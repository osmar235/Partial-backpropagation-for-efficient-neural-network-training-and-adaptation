"""Artificial invariants; no real checkpoint scoring or training."""
import argparse
import copy
import json
import unittest
import torch
import stage14_optimizer_history as r
from review_stage14C import assess

DEVICE=torch.device('cpu')


def spec():
    s=json.loads(r.SPEC.read_text());s.update(source_optimizer_age=2,after_blocks=1)
    return s


class Tests(unittest.TestCase):
    def test_both_histories_restore_weights_and_discard_warmup(self):
        s=spec();template,cp=r.source(9961,True,{})
        original=copy.deepcopy(cp)
        x=torch.randn(3,3,32,32,device=DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        for mode in ('full','spatial','similarity'):
            initial=[]
            for history in ('restore','fresh'):
                core=copy.deepcopy(template);opt=r.restore(core,cp,s,DEVICE,history)
                initial_hash=r.optimizer_hash(opt);model=r.t.TokenModel(core,mode,1).eval()
                with torch.no_grad():initial.append(model(x).detach().clone())
                assert len(opt.state)==(len(list(core.parameters())) if history=='restore' else 0)
                model.train();r.old.update(model,opt,x,y,131,.1)
                age=3 if history=='restore' else 1
                assert all(int(opt.state[p]['step'])==age for p in core.parameters())
                opt=r.restore(core,cp,s,DEVICE,history)
                assert r.old.state_hash(core.state_dict())==r.old.state_hash(cp['model_state'])
                assert r.optimizer_hash(opt)==initial_hash
                control=copy.deepcopy(template);other=r.restore(control,cp,s,DEVICE,history)
                ref=r.t.TokenModel(control,mode,1).train()
                assert r.old.update(model,opt,x,y,132,.1)==r.old.update(ref,other,x,y,132,.1)
                assert r.old.state_hash(core.state_dict())==r.old.state_hash(control.state_dict())
                assert r.optimizer_hash(opt)==r.optimizer_hash(other)
            assert torch.equal(*initial)
        assert r.old.state_hash(cp['model_state'])==r.old.state_hash(original['model_state'])
        for i,state in cp['optimizer_state']['state'].items():
            for key,value in state.items():assert torch.equal(value,original['optimizer_state']['state'][i][key])

    def test_full_wrapper_native_and_previous_restoration(self):
        import stage14_recovery as previous
        s=spec();a,cp=r.source(9961,True,{});b=copy.deepcopy(a)
        oa=r.restore(a,cp,s,DEVICE);ob=previous.restore(b,cp,s,DEVICE)
        assert r.optimizer_hash(oa)==r.optimizer_hash(ob)
        x=torch.randn(3,3,32,32,device=DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        a.train();wrap=r.t.TokenModel(b,'full',1).train()
        for step in range(3):
            for opt in (oa,ob):opt.param_groups[0]['lr']=1e-4
            assert r.old.update(a,oa,x,y,81+step,.1)==r.old.update(wrap,ob,x,y,81+step,.1)
            assert r.old.state_hash(a.state_dict())==r.old.state_hash(b.state_dict())
            assert r.optimizer_hash(oa)==r.optimizer_hash(ob)

    def test_invalid_mapping_age_rejected_even_for_fresh(self):
        s=spec();core,cp=r.source(9961,True,{})
        for history in ('restore','fresh'):
            bad=copy.deepcopy(cp);bad['optimizer_state']['state'][0]['step'].add_(1)
            with self.assertRaises(AssertionError):r.restore(core,bad,s,DEVICE,history)
            bad=copy.deepcopy(cp);bad['optimizer_state']['param_groups'][0]['params'].reverse()
            with self.assertRaises(AssertionError):r.restore(core,bad,s,DEVICE,history)

    def test_native_shape_all_six_arms_and_diagnostics(self):
        core=r.m.make_model('full_rgb32',81,False).to(DEVICE)
        with torch.no_grad():core.head.weight.normal_(std=.02)
        opt=torch.optim.AdamW(core.parameters(),lr=.001,weight_decay=.05)
        x=torch.randn(2,3,32,32,device=DEVICE);y=torch.tensor([1,2],device=DEVICE)
        core.train();r.old.update(core,opt,x,y,83,.1)
        cp=dict(model_state=copy.deepcopy(core.state_dict()),optimizer_state=copy.deepcopy(opt.state_dict()),updates=1)
        s=spec();s['source_optimizer_age']=1
        for arm in s['modes']:
            mode,history=arm.rsplit('_',1)
            opt=r.restore(core,cp,s,DEVICE,history);model=r.t.TokenModel(core,mode,3).train()
            before=r.diagnostic_state(core,opt)
            if history=='fresh':assert before['moment1_l2']==before['moment2_l2']==0
            r.old.update(model,opt,x,y,84,.1)
            assert all(int(opt.state[p]['step'])==(2 if history=='restore' else 1) for p in core.parameters())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters())
            assert (core.body.encoder.pos_embedding.grad.abs().sum(-1)>0).all()
            after=r.diagnostic_state(core,opt)
            assert float((after['weights']-before['weights']).norm())>0
            assert after['moment1_l2']>0 and after['moment2_l2']>0

    def test_paired_streams_and_lr_unchanged(self):
        s=spec();s.update(expected_batches_per_epoch=6,epochs=5)
        x=torch.rand(3,3,32,32,device=DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        states=[]
        for _ in s['modes']:
            bg,ag,mg,pg=r.generators(1501,s)
            batches=list(r.v.frozen6.epoch_batches(16,3,bg));assert sum(map(len,batches))==16
            xx=r.v.frozen6.augment_and_normalize(x,ag)
            xx,yy,lam=r.v.mix_batch(xx,y,mg,pg,.2,10)
            states.append((r.tensor_hash(xx),r.tensor_hash(yy),lam,r.tensor_hash(ag.get_state()),r.tensor_hash(pg.get_state())))
        assert all(state==states[0] for state in states)
        rates=[r.v.learning_rate(i,30,6,1e-4,s) for i in range(30)]
        assert abs(rates[0]-1e-4/6)<1e-15 and rates[5]==rates[6]==1e-4
        assert abs(rates[-1]-1e-5)<1e-15

    def test_quality_anchor_fastest_control_and_time_only(self):
        s=spec()
        rows=[dict(seed=seed,mode=mode,initial_accuracy=.82,final_accuracy=.82,
            complete_adaptation_seconds=100 if mode.startswith('full') else 90)
            for seed in s['seeds'] for mode in s['modes']]
        # Spatial is equally good and just as fast: no practical win.
        assert not assess(rows,s,False)[1]['practical']
        for row in rows:
            if row['mode'].startswith('spatial'):row['final_accuracy']=.81
            if row['mode']=='similarity_fresh':row['final_accuracy']=.817
        assert assess(rows,s,False)[1]['practical']
        rows[-1]['final_accuracy']=.8169
        assert not assess(rows,s,False)[1]['practical']
        for row in rows:
            row['final_accuracy']=.80
        assert not assess(rows,s,False)[1]['practical']
        for row in rows:
            row['final_accuracy']=.82
            if row['mode']=='similarity_fresh':row['complete_adaptation_seconds']=110
        assert not assess(rows,s,False)[1]['practical']
        assert not any(assess(rows,s,True)[1].values())

    def test_interaction_requires_absolute_gain_both_seeds(self):
        s=spec()
        rows=[dict(seed=seed,mode=mode,initial_accuracy=.82,final_accuracy=.82,complete_adaptation_seconds=100)
            for seed in s['seeds'] for mode in s['modes']]
        for row in rows:
            if row['mode']=='similarity_fresh':row['final_accuracy']=.825
        assert assess(rows,s,False)[1]['mechanism']
        for row in rows:
            if row['mode']=='full_fresh':row['final_accuracy']=.825
        assert not assess(rows,s,False)[1]['mechanism']
        for row in rows:
            if row['mode']=='full_fresh':row['final_accuracy']=.81
            if row['mode']=='similarity_fresh':row['final_accuracy']=.82
        assert not assess(rows,s,False)[1]['mechanism']
        for row in rows:
            if row['mode']=='similarity_fresh' and row['seed']==s['seeds'][0]:row['final_accuracy']=.83
        assert not assess(rows,s,False)[1]['mechanism']


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');r.m.configure()
    unittest.main(argv=['test_stage14_optimizer_history'],verbosity=2)
