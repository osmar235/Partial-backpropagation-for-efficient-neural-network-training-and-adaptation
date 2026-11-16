"""Three-block supervised local losses with synchronous pre-update features."""
import torch
from torch import nn
import torch.nn.functional as F
import stage12_pretrained as p

MODES=['full','full_aux','local','suffix']
class LocalModel(nn.Module):
    def __init__(self,backbone,seed,fake=False):
        super().__init__();self.net=backbone
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+311001)
            self.aux0=nn.Linear(8 if fake else 128,10);self.aux1=nn.Linear(12 if fake else 256,10)
        self.eval()
    def block0(self,x):
        m=self.net;return m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))
    def block1(self,x):return self.net.layer3(x)
    def block2(self,x):return self.net.fc(self.net.avgpool(self.net.layer4(x)).flatten(1))
    def forward(self,x):return self.block2(self.block1(self.block0(x)))
    def auxiliary(self,z,index):return (self.aux0 if index==0 else self.aux1)(z.mean((2,3)))

def group(name):
    if name.startswith('aux'):return 'aux'
    if name.startswith('net.layer3.'):return 'block1'
    if name.startswith(('net.layer4.','net.fc.')):return 'block2'
    return 'block0'

def optimizer(model,spec):
    return torch.optim.AdamW([dict(params=[t for n,t in model.named_parameters() if not n.startswith(('aux','net.fc.'))],lr=spec['backbone_lr']),
        dict(params=[t for n,t in model.named_parameters() if n.startswith(('aux','net.fc.'))],lr=spec['head_lr'])],weight_decay=spec['weight_decay'])

def backward(model,x,y,mode):
    model.eval();model.zero_grad(set_to_none=True)
    if mode=='suffix':
        with torch.no_grad():z=model.block1(model.block0(x))
        final=F.cross_entropy(model.block2(z),y);final.backward();return [float(final.detach())]
    z0=model.block0(x)
    if mode=='local':
        a=F.cross_entropy(model.auxiliary(z0,0),y);a.backward();z0=z0.detach()
        z1=model.block1(z0);b=F.cross_entropy(model.auxiliary(z1,1),y);b.backward();z1=z1.detach()
        c=F.cross_entropy(model.block2(z1),y);c.backward()
        return [float(a.detach()),float(b.detach()),float(c.detach())]
    z1=model.block1(z0);c=F.cross_entropy(model.block2(z1),y)
    if mode=='full':c.backward();return [float(c.detach())]
    assert mode=='full_aux'
    a=F.cross_entropy(model.auxiliary(z0,0),y);b=F.cross_entropy(model.auxiliary(z1,1),y)
    (a+b+c).backward();return [float(a.detach()),float(b.detach()),float(c.detach())]

def step(model,opt,x,y,mode):
    losses=backward(model,x,y,mode)
    if not all(torch.isfinite(torch.tensor(losses))):raise FloatingPointError('Nonfinite local loss')
    opt.step();return losses

def gradient_vectors(model):
    return {key:torch.cat([(t.grad.detach() if t.grad is not None else torch.zeros_like(t)).flatten().cpu()
                          for n,t in model.named_parameters() if group(n)==key]) for key in ['block0','block1','block2']}

def zero_optimizer(opt):
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):value.zero_()

def active(name,mode):return mode in ['local','full_aux'] or (not name.startswith('aux') and (mode=='full' or group(name)=='block2'))

def verify_step(model,opt,initial,mode):
    for n,t in model.named_parameters():
        assert torch.isfinite(t).all()
        if active(n,mode):
            assert t.grad is not None and torch.isfinite(t.grad).all() and int(opt.state[t]['step'])==1
            for key in ['exp_avg','exp_avg_sq']:assert torch.isfinite(opt.state[t][key]).all()
        else:
            assert t.grad is None and t not in opt.state and torch.equal(t.detach().cpu(),initial[n]),n
    for n,t in model.named_buffers():assert torch.equal(t.cpu(),initial[n]),n
