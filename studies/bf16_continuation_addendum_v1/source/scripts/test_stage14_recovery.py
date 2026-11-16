"""Artificial scientific invariants for Stage14B; no real data/checkpoint scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
import unittest
import numpy as np
import torch
import stage14_recovery as r
from review_stage14B import assess

DEVICE = torch.device('cpu')


def spec():
    s=json.loads(r.SPEC.read_text()); s.update(source_optimizer_age=2,after_blocks=1)
    return s


def artificial():
    return r.source(9961,True,{})


class Tests(unittest.TestCase):
    def test_optimizer_restoration_and_warmup_isolation(self):
        s=spec(); template,cp=artificial(); original=copy.deepcopy(cp)
        x=torch.randn(3,3,32,32,device=DEVICE); y=torch.tensor([1,2,3],device=DEVICE)
        for mode in ('full','spatial','similarity'):
            core=copy.deepcopy(template); opt=r.restore(core,cp,s,DEVICE)
            model=r.t.TokenModel(core,mode,1).train()
            original_hash=r.optimizer_hash(opt)
            r.old.update(model,opt,x,y,131,.1)
            assert all(int(opt.state[p]['step'])==3 for p in core.parameters())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters())
            opt=r.restore(core,cp,s,DEVICE)
            assert r.old.state_hash(core.state_dict())==r.old.state_hash(cp['model_state'])
            assert r.optimizer_hash(opt)==original_hash
            control=copy.deepcopy(template); other=r.restore(control,cp,s,DEVICE)
            ref=r.t.TokenModel(control,mode,1).train()
            a=r.old.update(model,opt,x,y,132,.1); b=r.old.update(ref,other,x,y,132,.1)
            assert a==b and r.old.state_hash(core.state_dict())==r.old.state_hash(control.state_dict())
            assert r.optimizer_hash(opt)==r.optimizer_hash(other)
        assert r.old.state_hash(cp['model_state'])==r.old.state_hash(original['model_state'])
        for i,state in cp['optimizer_state']['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,original['optimizer_state']['state'][i][key],rtol=0,atol=0)

    def test_full_update_equals_native(self):
        s=spec(); a,cp=artificial(); b=copy.deepcopy(a)
        oa=r.restore(a,cp,s,DEVICE); ob=r.restore(b,cp,s,DEVICE)
        x=torch.randn(3,3,32,32,device=DEVICE); y=torch.tensor([1,2,3],device=DEVICE)
        a.train(); wrap=r.t.TokenModel(b,'full',1).train()
        assert r.old.update(a,oa,x,y,81,.1)==r.old.update(wrap,ob,x,y,81,.1)
        assert r.old.state_hash(a.state_dict())==r.old.state_hash(b.state_dict())
        assert r.optimizer_hash(oa)==r.optimizer_hash(ob)

    def test_bad_optimizer_age_or_mapping_rejected(self):
        s=spec(); core,cp=artificial()
        bad=copy.deepcopy(cp); bad['optimizer_state']['state'][0]['step'].add_(1)
        with self.assertRaises(AssertionError): r.restore(core,bad,s,DEVICE)
        bad=copy.deepcopy(cp); bad['optimizer_state']['param_groups'][0]['params'].reverse()
        with self.assertRaises(AssertionError): r.restore(core,bad,s,DEVICE)

    def test_native_shape_training_all_modes(self):
        # Full-size architecture with artificial weights, inputs and Adam state.
        core=r.m.make_model('full_rgb32',81,False).to(DEVICE)
        with torch.no_grad(): core.head.weight.normal_(std=.02)
        opt=torch.optim.AdamW(core.parameters(),lr=.001,weight_decay=.05)
        x=torch.randn(2,3,32,32,device=DEVICE); y=torch.tensor([1,2],device=DEVICE)
        core.train(); r.old.update(core,opt,x,y,83,.1)
        cp=dict(model_state=copy.deepcopy(core.state_dict()),optimizer_state=copy.deepcopy(opt.state_dict()),updates=1)
        s=spec(); s['source_optimizer_age']=1
        for mode in ('full','spatial','similarity'):
            opt=r.restore(core,cp,s,DEVICE); model=r.t.TokenModel(core,mode,3).train()
            r.old.update(model,opt,x,y,84,.1)
            assert all(int(opt.state[p]['step'])==2 for p in core.parameters())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters())
            assert (core.body.encoder.pos_embedding.grad.abs().sum(-1)>0).all()

    def test_paired_streams_and_schedule(self):
        s=spec(); s.update(expected_batches_per_epoch=6,epochs=5)
        x=torch.rand(3,3,32,32,device=DEVICE); y=torch.tensor([1,2,3],device=DEVICE)
        states=[]
        for _ in range(3):
            bg,ag,mg,pg=r.generators(1501,s)
            batches=list(r.v.frozen6.epoch_batches(16,3,bg)); assert sum(map(len,batches))==16
            xx=r.v.frozen6.augment_and_normalize(x,ag)
            xx,yy,lam=r.v.mix_batch(xx,y,mg,pg,.2,10)
            states.append((r.tensor_hash(xx),r.tensor_hash(yy),lam,r.tensor_hash(ag.get_state()),r.tensor_hash(pg.get_state())))
        assert states[0]==states[1]==states[2]
        _,ag,_,_=r.generators(1501,s)
        torch.randint(0,9,(3,),generator=ag);torch.randint(0,9,(3,),generator=ag);torch.rand(3,generator=ag)
        assert r.tensor_hash(ag.get_state())==states[0][3]
        rates=[r.v.learning_rate(i,30,6,1e-4,s) for i in range(30)]
        assert abs(rates[0]-1e-4/6)<1e-15 and rates[5]==rates[6]==1e-4
        assert abs(rates[-1]-1e-5)<1e-15 and all(rates[i]>=rates[i+1] for i in range(6,29))

    def test_screen_reference_anchor_and_simple_control(self):
        s=spec()
        rows=[dict(seed=seed,mode=mode,initial_accuracy=.82,final_accuracy=.82,
                   charged_recovery_seconds=100 if mode=='full' else 90,
                   max_training_incremental_allocated=1000 if mode=='full' else 800)
              for seed in s['seeds'] for mode in s['modes']]
        _,both,justified,decision=assess(rows,s,False)
        assert both==dict(spatial=True,similarity=True) and not justified and decision=='prefer_spatial'
        for row in rows:
            if row['mode']=='spatial': row['final_accuracy']=.81
            if row['mode']=='similarity': row['final_accuracy']=.817
        _,both,justified,_=assess(rows,s,False)
        assert justified and both['similarity'] and not both['spatial']
        rows[-1]['final_accuracy']=.8169
        assert not assess(rows,s,False)[1]['similarity']
        for row in rows: row['final_accuracy']=.80
        assert not any(assess(rows,s,False)[1].values())
        for row in rows: row['final_accuracy']=.82
        assert not any(assess(rows,s,True)[1].values())
        # Resource route may be memory-only; total time must not be called a speedup.
        for row in rows:
            if row['mode']!='full': row['charged_recovery_seconds']=110
        screens,both,_,_=assess(rows,s,False)
        assert both['similarity'] and all(row['memory_pass'] and not row['time_pass'] for row in screens)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cuda',action='store_true'); args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu'); r.m.configure()
    unittest.main(argv=['test_stage14_recovery'],verbosity=2)
