"""Exact linear forward/input derivative; sampled MLP weight outer products.

First-order FP32 prototype. The only approximated quantity is dW. All bias,
input and other module gradients retain ordinary backpropagation. No novelty
claim: column-row sampling and sampled backpropagation are established work.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

class WeightGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,indices,factors):
        ctx.shape=x.shape;ctx.has_bias=bias is not None
        ctx.save_for_backward(x.reshape(-1,x.shape[-1]).index_select(0,indices),weight,indices,factors)
        return F.linear(x,weight,bias)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        selected_x,weight,indices,factors=ctx.saved_tensors
        g=grad.reshape(-1,grad.shape[-1])
        dx=(g@weight).reshape(ctx.shape) if ctx.needs_input_grad[0] else None
        dw=(g.index_select(0,indices)*factors[:,None]).T@selected_x if ctx.needs_input_grad[1] else None
        db=g.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx,dw,db,None,None

class SampledWeightLinear(nn.Module):
    def __init__(self,linear,patches_kept,layer_index):
        super().__init__();self.weight=linear.weight;self.bias=linear.bias
        self.patches_kept=patches_kept;self.layer_index=layer_index;self.sampling_seed=0

    def forward(self,x):
        if not torch.is_grad_enabled() or not self.training:return F.linear(x,self.weight,self.bias)
        assert x.ndim==3 and 0<self.patches_kept<=x.shape[1]-1
        batch,tokens,_=x.shape;patches=tokens-1
        # One uniform spatial subset per layer/update, shared across images.
        # A private generator cannot consume the model's dropout stream.
        gen=torch.Generator(device=x.device).manual_seed(self.sampling_seed+1009*self.layer_index)
        positions=torch.cat((torch.zeros(1,device=x.device,dtype=torch.long),
            1+torch.randperm(patches,generator=gen,device=x.device)[:self.patches_kept]))
        indices=(torch.arange(batch,device=x.device)[:,None]*tokens+positions[None,:]).reshape(-1)
        factors=torch.full((batch,len(positions)),patches/self.patches_kept,device=x.device,dtype=x.dtype)
        factors[:,0]=1
        return WeightGradientFunction.apply(x,self.weight,self.bias,indices,factors.reshape(-1))

def install(model,patches_kept):
    count=0
    for block in model.body.encoder.layers:
        for index,layer in enumerate(block.mlp):
            if isinstance(layer,nn.Linear):
                block.mlp[index]=SampledWeightLinear(layer,patches_kept,count);count+=1
    assert count==2*len(model.body.encoder.layers)
    return model

def set_sampling_seed(model,seed):
    for layer in model.modules():
        if isinstance(layer,SampledWeightLinear):layer.sampling_seed=seed
