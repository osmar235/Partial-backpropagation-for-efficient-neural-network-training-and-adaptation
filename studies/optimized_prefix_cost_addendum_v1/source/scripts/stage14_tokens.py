"""Single-boundary token merging with preserved CLS and proportional attention.

ToMe-inspired diagnostic, not a reproduction: match residual features once
after a complete block, rather than attention keys at multiple sublayers.
"""
import torch
from torch import nn
import torch.nn.functional as F

MODES = ('full', 'spatial', 'similarity')


def merge_tokens(x, mode):
    """Merge all 32 even patch tokens into the 32 odd patch destinations.

    Routing is discrete and detached. Weighted means retain gradients to every
    source token. Dense one-hot bmm avoids nondeterministic scatter reductions.
    Raster order makes spatial pairs horizontal neighbors on the 8x8 grid.
    """
    assert mode in ('spatial', 'similarity') and x.shape[1] == 65
    cls, patches = x[:, :1], x[:, 1:]
    left, right = patches[:, ::2], patches[:, 1::2]
    with torch.no_grad():
        if mode == 'spatial':
            destination = torch.arange(32, device=x.device).expand(len(x), -1)
        else:
            a, b = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
            destination = (a @ b.transpose(1, 2)).argmax(-1)
        assignment = F.one_hot(destination, 32).to(x.dtype)
        mass = 1 + assignment.sum(dim=1)
    merged = (right + assignment.transpose(1, 2) @ left) / mass.unsqueeze(-1)
    return torch.cat((cls, merged), 1), torch.cat((mass.new_ones(len(x), 1), mass), 1)


def proportional_block(block, x, mass):
    """Original block weights; log group size biases keys in every head."""
    batch, length, _ = x.shape
    bias = mass.log()[:, None, None, :].expand(batch, block.num_heads, length, length)
    bias = bias.reshape(batch * block.num_heads, length, length)
    z = block.ln_1(x)
    z, _ = block.self_attention(z, z, z, attn_mask=bias, need_weights=False)
    residual = x + block.dropout(z)
    return residual + block.mlp(block.ln_2(residual))


class TokenModel(nn.Module):
    def __init__(self, core, mode, after_blocks):
        super().__init__()
        assert mode in MODES and 0 < after_blocks < len(core.body.encoder.layers)
        assert core.variant == 'full_rgb32'
        self.core, self.mode, self.after_blocks = core, mode, after_blocks

    def forward(self, images):
        if self.mode == 'full':
            return self.core(images)
        body = self.core.body
        x = body._process_input(images)
        x = torch.cat((body.class_token.expand(len(x), -1, -1), x), 1)
        x = body.encoder.dropout(x + body.encoder.pos_embedding)
        mass = None
        for i, block in enumerate(body.encoder.layers):
            x = block(x) if mass is None else proportional_block(block, x, mass)
            if i + 1 == self.after_blocks:
                x, mass = merge_tokens(x, self.mode)
        return self.core.head(body.encoder.ln(x)[:, 0])
