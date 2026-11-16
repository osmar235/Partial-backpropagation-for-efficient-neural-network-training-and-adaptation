"""Lossless Haar/pixel representations and fully differentiable comparison models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import torch
from torch import nn
import torch.nn.functional as F
import stage8_vit_v2 as v

VARIANTS=['full_rgb32','small_rgb32','coarse_rgb16','pixel_branches','haar_branches']

def configure():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Start with CUBLAS_WORKSPACE_CONFIG=:4096:8'
    torch.set_num_threads(1);v.configure();torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)

def execution_settings():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],cpu_threads=torch.get_num_threads(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)

def phases(x):
    assert x.ndim==4 and x.shape[-1]%2==x.shape[-2]%2==0
    return x[:,:,0::2,0::2],x[:,:,0::2,1::2],x[:,:,1::2,0::2],x[:,:,1::2,1::2]

def haar(x):
    a,b,c,d=phases(x)
    # LL=2*mean ensures model LL/2 is exactly the ordinary avg-pool control.
    ll=2*F.avg_pool2d(x,2)
    details=torch.cat([.5*(a-b+c-d),.5*(a+b-c-d),.5*(a-b-c+d)],dim=1)
    return ll,details

def inverse_haar(ll,details):
    h,vv,d=details.chunk(3,dim=1)
    a=.5*(ll+h+vv+d);b=.5*(ll-h+vv-d);c=.5*(ll+h-vv-d);e=.5*(ll-h-vv+d)
    out=ll.new_empty(ll.shape[0],ll.shape[1],ll.shape[2]*2,ll.shape[3]*2)
    out[:,:,0::2,0::2]=a;out[:,:,0::2,1::2]=b;out[:,:,1::2,0::2]=c;out[:,:,1::2,1::2]=e
    return out

def pixel_groups(x):
    a,b,c,d=phases(x);return a.contiguous(),torch.cat([b,c,d],dim=1)

class RepresentationModel(nn.Module):
    def __init__(self,variant,seed,fake=False):
        super().__init__();assert variant in VARIANTS;self.variant=variant
        low=variant in ['coarse_rgb16','pixel_branches','haar_branches'];small=variant=='small_rgb32'
        spec=copy.deepcopy(v.SPEC);spec['architecture']['image_size']=16 if low else 32
        if small:spec['architecture'].update(hidden_dim=96,mlp_dim=384)
        if fake:
            width=12 if small else 24
            spec['architecture'].update(num_layers=2,num_heads=3,hidden_dim=width,mlp_dim=2*width)
        width=spec['architecture']['hidden_dim']
        # Component seeds are independent of construction/order of other branches.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+181001);self.body=v.make_model(spec);self.body.heads=nn.Identity()
        self.dual=variant in ['pixel_branches','haar_branches']
        detail_width=8 if fake else 48;middle=4 if fake else 24
        if self.dual:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+182001)
                self.detail=nn.Sequential(nn.Conv2d(9,middle,3,stride=2,padding=1),nn.GELU(),
                    nn.Conv2d(middle,detail_width,3,stride=2,padding=1),nn.GELU())
                self.detail_norm=nn.LayerNorm(detail_width)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+183001);self.head=nn.Linear(width+(detail_width if self.dual else 0),10)
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x):
        detail=None
        if self.variant=='coarse_rgb16':coarse=F.avg_pool2d(x,2)
        elif self.variant=='haar_branches':
            ll,detail=haar(x);coarse=.5*ll
        elif self.variant=='pixel_branches':coarse,detail=pixel_groups(x)
        else:coarse=x
        representation=self.body(coarse)
        if detail is not None:
            detail_features=self.detail_norm(self.detail(detail).mean(dim=(-1,-2)))
            representation=torch.cat([representation,detail_features],dim=1)
        return self.head(representation)

def make_model(variant,seed,fake=False):return RepresentationModel(variant,seed,fake)
