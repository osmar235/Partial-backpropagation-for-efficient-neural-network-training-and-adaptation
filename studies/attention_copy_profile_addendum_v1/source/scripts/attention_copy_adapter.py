"""Diagnostic native-layout control; no equivalence or novelty assumed."""
from contextlib import nullcontext
import torch
from attention_reference_adapter import ReferenceSelfAttention


class ProfileAttention(ReferenceSelfAttention):
    def __init__(self, original, keep_copy, index):
        super().__init__(original)
        self.keep_copy=keep_copy;self.index=index;self.profile_enabled=False

    def forward(self,query,key,value,key_padding_mask=None,need_weights=True,attn_mask=None,
                average_attn_weights=True,is_causal=False):
        assert query is key and key is value
        assert query.ndim==3 and query.shape[-1]==self.embed_dim
        assert key_padding_mask is None and attn_mask is None and not is_causal and not need_weights
        output=self._reference(query)
        if self.keep_copy:
            if self.profile_enabled:
                with torch.profiler.record_function(f'module::compatibility_copy_{self.index}'):
                    output=output.transpose(0,1).contiguous().transpose(0,1)
            else:
                output=output.transpose(0,1).contiguous().transpose(0,1)
        return output,None


def install(model,arm):
    assert arm in ('compatible','native_layout_control')
    ids={n:id(p) for n,p in model.named_parameters()};names=set(model.state_dict())
    for i,block in enumerate(model.body.encoder.layers):
        block.self_attention=ProfileAttention(block.self_attention,arm=='compatible',i)
    assert ids=={n:id(p) for n,p in model.named_parameters()} and names==set(model.state_dict())
    return model


def enable_ranges(model,enabled):
    for block in model.body.encoder.layers:block.self_attention.profile_enabled=enabled
