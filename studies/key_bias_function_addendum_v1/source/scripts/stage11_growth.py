"""Width doubling for the existing ViT, retaining three attention heads.

Preserves the dropout-free function, not individual stochastic masks or Adam
trajectories. The optimizer is deliberately reset, never silently transplanted.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import stage10_models as m


def lift_state(state, heads=3):
    """Differentiable linear map of every parameter; no unidentified defaults."""
    width = state['body.class_token'].shape[-1]
    assert width % heads == 0
    device = state['body.class_token'].device
    embed = torch.arange(width, device=device).repeat(2)
    # Attention channels repeat WITHIN each head, unlike residual embeddings.
    attn = torch.arange(width, device=device).reshape(heads, -1).repeat(1, 2).flatten()
    qkv = torch.cat([attn + j * width for j in range(3)])
    qk_scale = 2 ** (-.25)
    lifted = {}
    for name, value in state.items():
        if name in ('body.class_token', 'body.encoder.pos_embedding'):
            out = value.index_select(-1, embed)
        elif name.startswith('body.conv_proj.'):
            out = value.index_select(0, embed)
        elif name == 'head.weight':
            out = value.index_select(1, embed) / 2
        elif name == 'head.bias':
            out = value.clone()
        elif name.startswith('body.encoder.ln.') or '.ln_1.' in name or '.ln_2.' in name:
            out = value.index_select(0, embed)
        elif name.endswith('self_attention.in_proj_weight'):
            out = value.index_select(0, qkv).index_select(1, embed) / 2
            scale = out.new_ones(6 * width, 1)
            scale[:4 * width] = qk_scale
            out = out * scale
        elif name.endswith('self_attention.in_proj_bias'):
            out = value.index_select(0, qkv)
            scale = out.new_ones(6 * width)
            scale[:4 * width] = qk_scale
            out = out * scale
        elif name.endswith('self_attention.out_proj.weight'):
            out = value.index_select(0, embed).index_select(1, attn) / 2
        elif name.endswith('self_attention.out_proj.bias'):
            out = value.index_select(0, embed)
        elif name.endswith('mlp.0.weight'):
            hidden = torch.arange(value.shape[0], device=device).repeat(2)
            out = value.index_select(0, hidden).index_select(1, embed) / 2
        elif name.endswith('mlp.0.bias'):
            out = value.repeat(2)
        elif name.endswith('mlp.3.weight'):
            hidden = torch.arange(value.shape[1], device=device).repeat(2)
            out = value.index_select(0, embed).index_select(1, hidden) / 2
        elif name.endswith('mlp.3.bias'):
            out = value.index_select(0, embed)
        else:
            raise ValueError('Unsupported source parameter: ' + name)
        lifted[name] = out
    return lifted


def expand(small, seed, fake=False):
    assert small.variant == 'small_rgb32' and not small.dual
    source = next(small.parameters())
    large = m.make_model('full_rgb32', seed, fake).to(device=source.device, dtype=source.dtype)
    lifted = lift_state(small.state_dict())
    expected = large.state_dict()
    assert lifted.keys() == expected.keys()
    assert all(lifted[k].shape == expected[k].shape for k in expected)
    large.load_state_dict(lifted, strict=True)
    large.train(small.training)
    return large


def reset_adam(model, lr=.001, weight_decay=.05):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def disable_dropout(model):
    """For mathematical tests only; experiment training retains dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.


def twin_difference(model):
    weight = model.body.conv_proj.weight.detach()
    half = weight.shape[0] // 2
    return float((weight[:half] - weight[half:]).abs().max())
