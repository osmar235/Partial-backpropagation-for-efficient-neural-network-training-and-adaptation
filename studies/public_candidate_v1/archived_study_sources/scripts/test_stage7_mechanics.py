"""Fast unit tests of Stage7 scheduling, autograd, Adam moment/None behaviors."""
from __future__ import annotations
import importlib.util
import math
import copy
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn

MODULE = Path(__file__).with_name('stage7_review_experiments.py')
spec = importlib.util.spec_from_file_location('stage7_review_experiments', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def check_schedules():
    for n in (2,8,50,5880):
        for randomized in (False, True):
            s = m.matched_partial_indices(n,m.MATCHED_Q,51,randomized)
            assert len(s) == round(n*m.MATCHED_Q), (n, randomized, len(s))
            assert 0 not in s
            assert all(0<=v<n and v+1 not in s for v in s)
            assert s == m.matched_partial_indices(n,m.MATCHED_Q,51,randomized)


def check_epochwise_matched():
    for randomize in (False,True):
        plan = m.epoch_matched_partial_indices(30,196,51,randomize)
        assert len(plan) == sum(round(q*196) for q in m.MATCHED_EPOCH_Q) == 2116
        assert all(j+1 not in plan for j in plan)
        assert all((epoch*196) not in plan for epoch in range(30))

def check_suffix_exactness():
    torch.manual_seed(112)
    model = m.TinyCifar(10)
    xb = torch.randn(2,3,8,8)
    yb = torch.tensor([0,1])
    _, suffix = m.partition(model,'lgpb')
    out = nn.CrossEntropyLoss()(model(xb),yb)
    full = torch.autograd.grad(out,list(model.parameters()),retain_graph=True)
    selected = torch.autograd.grad(out,suffix)
    mapping = {id(p):g for p,g in zip(model.parameters(),full)}
    for p,g in zip(suffix,selected):
        assert torch.allclose(g, mapping[id(p)],atol=1e-6,rtol=1e-5)


def check_zero_vs_none_grad():
    p = nn.Parameter(torch.tensor([1.0,-1.0]))
    opt = torch.optim.Adam([p],lr=.01)
    p.grad=torch.tensor([.5,-.5]);opt.step()
    before=p.detach().clone()
    old_step=int(opt.state[p]['step'].item())
    p.grad=torch.zeros_like(p);opt.step()
    assert int(opt.state[p]['step'].item())==old_step+1
    assert not torch.equal(p,before), 'prefix moment continuation must move when prior moment nonzero'
    freeze_before=p.detach().clone(); old_step=int(opt.state[p]['step'].item())
    p.grad=None;opt.step()
    assert torch.equal(p,freeze_before), 'None grad freezes prefix parameters'
    assert int(opt.state[p]['step'].item())==old_step, 'None grad does not advance Adam step'


def check_gate():
    assert not m.choose_partial('lgpb',0,None,False,1.0,set())
    assert m.choose_partial('lgpb',1,1.1,False,1.0,set())
    assert not m.choose_partial('lgpb',2,1.1,True,1.0,set())
    assert not m.choose_partial('lgpb',3,1.1,False,1.2,set())
    assert m.choose_partial('no_forced_full',2,1.1,True,1.0,set())


def check_frozen_trajectory():
    """Compare every final weight, losses and skip counts to unchanged Stage 6."""
    torch.set_num_threads(1)
    torch.manual_seed(123)
    device = torch.device('cpu')
    model = m.TinyCifar(10)
    init = copy.deepcopy(model.state_dict())
    args = SimpleNamespace(seed=50, dataset='fake', num_classes=10, train_subset=64,
        eval_subset=8, optimizer='adam', lr=.0003, rho=.95, epochs=2,
        batch_size=8, schedule_epochs=2, min_lr_ratio=.01, log_steps=True,
        warmup_steps=0, eval_batch_size=8, match_policy='epoch', matched_q=m.MATCHED_Q)
    xtr,ytr,xev,yev = m.load_data(args,device)
    _,_,actual = m.run_one(args,'lgpb',model,init,xtr,ytr,xev,yev,device)
    actual_state = copy.deepcopy(model.state_dict())
    _,expected = m.frozen.run_one('loss_gated_moment_continuation_adam',args.lr,
        model,init,xtr,ytr,xev,yev,nn.CrossEntropyLoss(),device,args.epochs,
        args.schedule_epochs,args.batch_size,args.seed,args.eval_batch_size,
        args.min_lr_ratio,args.rho,(),0)
    assert actual['partial_updates'] > 0
    assert actual['partial_updates'] == expected['partial_updates']
    assert actual['final_eval_loss'] == expected['final_eval_loss']
    for name,value in model.state_dict().items():
        assert torch.equal(value,actual_state[name]), name
    print('OK: complete two-epoch trajectory matches frozen Stage 6 exactly, including partial steps')


def check_transfer_and_replay():
    args = SimpleNamespace(tiny_model=False,num_classes=100)
    model = m.make_model(args,torch.device('cpu'))
    assert model(torch.zeros(2,3,32,32)).shape == (2,100)
    augmented=m.augment(torch.rand(2,3,32,32),torch.Generator().manual_seed(50),'cifar100')
    assert augmented.shape == (2,3,32,32) and torch.isfinite(augmented).all()
    prefix,_=m.partition(model,'lgpb')
    before=[p.detach().clone() for p in prefix]
    delta=[torch.full_like(p,.001) for p in prefix]
    m.add_replayed_delta(prefix,delta)
    for p,b,d in zip(prefix,before,delta):
        assert torch.equal(p,b+d)
    print('OK: 100-class ResNet, CIFAR-100 augmentation, displacement replay')

if __name__=='__main__':
    check_schedules();check_epochwise_matched();check_suffix_exactness();check_zero_vs_none_grad();check_gate()
    check_frozen_trajectory()
    check_transfer_and_replay()
    print('OK: exact matched schedules, suffix gradient, Adam zero-vs-None, loss gate')
