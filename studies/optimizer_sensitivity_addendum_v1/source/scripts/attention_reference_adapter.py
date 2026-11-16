"""Native parameter interface around the exact pinned timm Attention class.

Only the class AST is executed from the archived upstream module. The upstream
forward is unchanged; use_fused_attn is fixed True so the caller selects SDPA.
No timm package installation or full-model architecture substitution is needed.
"""
import ast
import hashlib
import json
from pathlib import Path
from typing import Final
import torch
from torch import nn
import torch.nn.functional as F
from qkv_unbind import UnbindSelfAttention

UPSTREAM = Path(__file__).resolve().parents[1] / 'third_party/timm_attention_v0_9_2'

def load_attention_class():
    provenance = json.loads((UPSTREAM / 'PROVENANCE.json').read_text())
    for name, record in provenance['files'].items():
        assert hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() == record['sha256']
    path = UPSTREAM / 'vision_transformer.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Attention']
    assert len(nodes) == 1
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'Final': Final, 'use_fused_attn': lambda: True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Attention']

Attention = load_attention_class()

class ReferenceSelfAttention(UnbindSelfAttention):
    def __init__(self, original):
        super().__init__(original)
        with torch.random.fork_rng(devices=[]):
            reference = Attention(self.embed_dim, num_heads=self.num_heads, qkv_bias=True,
                                  qk_norm=False, attn_drop=0., proj_drop=0.)
        reference.qkv.weight = self.in_proj_weight
        reference.qkv.bias = self.in_proj_bias
        reference.proj = self.out_proj
        assert not list(reference.buffers())
        # Avoid duplicate state-dict registrations; the wrapper owns all shared
        # Parameter objects. The reference has no additional parameters/buffers.
        assert {id(p) for p in reference.parameters()} == {id(p) for p in self.parameters()}
        object.__setattr__(self, '_reference', reference)
        reference.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, '_reference'):
            self._reference.train(mode)
        return self

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        assert query is key and key is value
        assert query.ndim == 3 and query.shape[-1] == self.embed_dim
        assert key_padding_mask is None and attn_mask is None and not is_causal and not need_weights
        # Native MHA returns a sequence-contiguous tensor viewed batch-first.
        # Match this boundary stride: downstream dropout otherwise maps the
        # same RNG stream to different logical elements. Charge this copy to
        # the reference arm; the upstream Attention.forward remains untouched.
        output = self._reference(query)
        return output.transpose(0, 1).contiguous().transpose(0, 1), None

def install(model):
    identities = {n: id(p) for n, p in model.named_parameters()}
    names = set(model.state_dict())
    for block in model.body.encoder.layers:
        block.self_attention = ReferenceSelfAttention(block.self_attention)
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    assert names == set(model.state_dict())
    return model
