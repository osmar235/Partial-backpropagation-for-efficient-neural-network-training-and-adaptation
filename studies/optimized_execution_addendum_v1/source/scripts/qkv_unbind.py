"""One layout/autograd change: split packed QKV with unbind, not three selects.

Narrow self-attention adapter for the existing ViT. Same sequence-first GEMMs,
packed contiguous copy, SDPA views and output projection as installed PyTorch.
All original Parameter objects and state-dict names are retained.
"""
import torch
from torch import nn
import torch.nn.functional as F

class UnbindSelfAttention(nn.Module):
    def __init__(self,original):
        super().__init__()
        assert original.batch_first and original._qkv_same_embed_dim
        assert original.bias_k is None and original.bias_v is None and not original.add_zero_attn
        assert original.dropout==0 and original.in_proj_bias is not None
        self.embed_dim=original.embed_dim;self.num_heads=original.num_heads;self.dropout=original.dropout
        self.in_proj_weight=original.in_proj_weight;self.in_proj_bias=original.in_proj_bias
        self.out_proj=original.out_proj
        self.train(original.training)

    def forward(self,query,key,value,key_padding_mask=None,need_weights=True,attn_mask=None,average_attn_weights=True,is_causal=False):
        assert query is key and key is value,'Only self-attention is supported'
        assert query.ndim==3 and query.shape[-1]==self.embed_dim
        assert key_padding_mask is None and attn_mask is None and not is_causal and not need_weights
        x=query.transpose(1,0);length,batch,width=x.shape;head=width//self.num_heads
        packed=F.linear(x,self.in_proj_weight,self.in_proj_bias)
        packed=packed.unflatten(-1,(3,width)).unsqueeze(0).transpose(0,-2).squeeze(-2).contiguous()
        q,k,v=packed.unbind(0)
        q=q.view(length,batch*self.num_heads,head).transpose(0,1).view(batch,self.num_heads,length,head)
        k=k.view(length,batch*self.num_heads,head).transpose(0,1).view(batch,self.num_heads,length,head)
        v=v.view(length,batch*self.num_heads,head).transpose(0,1).view(batch,self.num_heads,length,head)
        output=F.scaled_dot_product_attention(q,k,v,None,0.0,False)
        output=output.permute(2,0,1,3).contiguous().view(batch*length,width)
        output=F.linear(output,self.out_proj.weight,self.out_proj.bias).view(length,batch,width)
        return output.transpose(1,0),None

def install(model):
    before={n:id(p) for n,p in model.named_parameters()};state=set(model.state_dict())
    for block in model.body.encoder.layers:block.self_attention=UnbindSelfAttention(block.self_attention)
    assert before=={n:id(p) for n,p in model.named_parameters()}
    assert state==set(model.state_dict())
    return model
