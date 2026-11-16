"""Pretrained ResNet transfer primitives; frozen prefixes receive no Adam update."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,hashlib,json,urllib.request
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights
import stage8_vit_v2 as v
import stage10_models as execution

ROOT=v.ROOT
WEIGHTS=ROOT/'data/pretrained/resnet18-f37072fd.pth'
URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare_weights():
    assert ResNet18_Weights.IMAGENET1K_V1.url==URL
    WEIGHTS.parent.mkdir(exist_ok=True)
    if not WEIGHTS.exists():
        pending=WEIGHTS.with_suffix('.download')
        with urllib.request.urlopen(URL,timeout=90) as response,pending.open('wb') as f:
            while chunk:=response.read(1024*1024):f.write(chunk)
        assert sha(pending).startswith('f37072fd'),'Official weight hash mismatch'
        pending.replace(WEIGHTS)
    digest=sha(WEIGHTS);assert digest.startswith('f37072fd')
    model=resnet18(weights=None);model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    record=dict(url=URL,enum='ResNet18_Weights.IMAGENET1K_V1',sha256=digest,bytes=WEIGHTS.stat().st_size,
                upstream_hash_prefix='f37072fd',classification_source='ImageNet-1K',download_does_not_train=True)
    metadata=WEIGHTS.with_suffix('.json')
    if metadata.exists():assert json.loads(metadata.read_text())==record
    else:v.atomic_json(metadata,record)
    print(json.dumps(record,indent=2))

class TinyResNet(nn.Module):
    """Artificial workflow only; native ResNet is tested separately."""
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(3,4,3,padding=1,bias=False);self.bn1=nn.BatchNorm2d(4)
        self.relu=nn.ReLU();self.maxpool=nn.Identity()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU())
        self.layer1=block(4,4);self.layer2=block(4,8);self.layer3=block(8,12);self.layer4=block(12,16)
        self.avgpool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(16,10)

def make_model(seed,fake=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(712001)
        model=TinyResNet() if fake else resnet18(weights=None)
        if not fake:model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
        torch.random.default_generator.manual_seed(seed+271001)
        model.fc=nn.Linear(model.fc.in_features,10)
    # BatchNorm statistics stay fixed for every arm; affine parameters may train.
    model.eval();return model

def groups(model):
    result={'prefix':[],'suffix':[],'head':[]}
    for name,p in model.named_parameters():result['head' if name.startswith('fc.') else 'suffix' if name.startswith('layer4.') else 'prefix'].append(p)
    return result

def forward(model,x,mode):
    assert mode in ['full','suffix','head']
    def prefix(x):return model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))))
    if mode=='full':z=prefix(x)
    else:
        with torch.no_grad():z=prefix(x)
    if mode=='head':
        with torch.no_grad():z=model.layer4(z)
    else:z=model.layer4(z)
    return model.fc(model.avgpool(z).flatten(1))

def optimizer(model,spec):
    gg=groups(model)
    return torch.optim.AdamW([dict(params=gg[k],lr=spec['head_lr'] if k=='head' else spec['backbone_lr'],role=k)
                              for k in ['prefix','suffix','head']],weight_decay=spec['weight_decay'])

def mode_at(arm,index,spec):
    warm=spec['head_warmup_epochs']*spec['batches_per_epoch']
    if arm=='full':return 'full'
    if arm=='head' or index<warm:return 'head'
    if arm=='lp_full':return 'full'
    if arm=='lp_suffix':return 'suffix'
    assert arm=='lp_periodic'
    return 'full' if (index-warm)%spec['prefix_period']==0 else 'suffix'

def rate_factor(index,spec):
    import math
    warm=spec['lr_warmup_epochs']*spec['batches_per_epoch'];total=spec['epochs']*spec['batches_per_epoch']
    if index<warm:return (index+1)/max(1,warm)
    t=(index-warm)/max(1,total-warm-1)
    return spec['min_lr_ratio']+(1-spec['min_lr_ratio'])*.5*(1+math.cos(math.pi*t))

def update(model,opt,x,y,mode,smoothing):
    model.eval();opt.zero_grad(set_to_none=True)
    loss=F.cross_entropy(forward(model,x,mode),y,label_smoothing=smoothing)
    value=loss.item()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation loss')
    loss.backward();opt.step();return value

def transform(x,size,generator=None):
    # CIFAR crop/flip occurs at native resolution, then bilinear resize and
    # ImageNet normalization. This 96px budget screen is not the224px recipe.
    if generator is not None:
        n=len(x);padded=F.pad(x,(4,4,4,4),mode='reflect')
        offsets=torch.randint(0,9,(n,2),generator=generator)
        flip=torch.rand(n,generator=generator)<.5
        base=torch.arange(32,device=x.device)
        rows=offsets[:,0].to(x.device)[:,None]+base[None,:]
        cols=offsets[:,1].to(x.device)[:,None]+base[None,:]
        x=padded.permute(0,2,3,1)[torch.arange(n,device=x.device)[:,None,None],rows[:,:,None],cols[:,None,:]].permute(0,3,1,2)
        x=torch.where(flip.to(x.device)[:,None,None,None],x.flip(-1),x)
    x=F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False,antialias=True)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def load_data(device,fake=False):
    if fake:
        gen=torch.Generator().manual_seed(712002)
        return (torch.rand(16,3,32,32,generator=gen).to(device),torch.randint(0,10,(16,),generator=gen).to(device),
                torch.rand(8,3,32,32,generator=gen).to(device),torch.randint(0,10,(8,),generator=gen).to(device),{'kind':'artificial'})
    from torchvision.datasets import CIFAR10
    dataset=CIFAR10(str(ROOT/'data'),train=True,download=False)
    x=torch.from_numpy(dataset.data).permute(0,3,1,2).contiguous().float()/255;y=torch.tensor(dataset.targets)
    tr,va=v.split_indices(dataset.targets)
    split=dict(kind='train_only_stratified_45000_5000',train_index_sha256=hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
               validation_index_sha256=hashlib.sha256(va.numpy().tobytes()).hexdigest())
    return x[tr].to(device),y[tr].to(device),x[va].to(device),y[va].to(device),split

@torch.no_grad()
def evaluate(model,x,y,spec):
    model.eval();loss=correct=0
    for i in range(0,len(y),spec['eval_batch_size']):
        xx=transform(x[i:i+spec['eval_batch_size']],spec['image_size']);yy=y[i:i+spec['eval_batch_size']]
        logits=forward(model,xx,'full');loss+=F.cross_entropy(logits,yy,reduction='sum').item();correct+=(logits.argmax(1)==yy).sum().item()
    return loss/len(y),correct/len(y)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-weights',action='store_true',required=True);ap.parse_args();prepare_weights()
