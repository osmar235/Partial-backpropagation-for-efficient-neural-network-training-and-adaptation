"""Artificial-only mechanism, transition, historical-control and decision checks."""
import argparse
import copy
import json
import unittest
import torch
import stage15_side_adaptation as r
from review_stage15A import assess

DEVICE=torch.device('cpu')

def spec():
    s=json.loads(r.SPEC.read_text());s.update(taps=[1,2],side_width=6,expected_batches_per_epoch=2,epochs=2,source_optimizer_age=2)
    return s

class SideTests(unittest.TestCase):
    def setUp(self):
        self.s=spec();self.seed=9971
        self.core,self.cp=r.source(self.seed,True,{})
        gen=torch.Generator().manual_seed(41)
        self.x=torch.randn(3,3,32,32,generator=gen).to(DEVICE);self.y=torch.tensor([0,1,2],device=DEVICE)

    def condition(self,mode):
        return r.make_condition(copy.deepcopy(self.core),self.cp,self.seed,mode,self.s,DEVICE)

    def test_initial_outputs_and_identical_side_states(self):
        models={mode:self.condition(mode)[0] for mode in self.s['modes']}
        self.assertEqual(r.old.state_hash(models['multi_side'].state_dict()),r.old.state_hash(models['final_side'].state_dict()))
        for training in (False,True):
            values={}
            for mode,model in models.items():
                model.train(training);torch.manual_seed(1701)
                with torch.no_grad():values[mode]=model(self.x)
            for mode in models:
                self.assertTrue(torch.equal(values['head'].argmax(1),values[mode].argmax(1)))
                self.assertLessEqual(float((values['full']-values[mode]).abs().max()),1e-5)
            for mode in ('lp_full','final_side','multi_side'):self.assertTrue(torch.equal(values['head'],values[mode]))

    def test_gradients_and_frozen_body(self):
        for mode in ('head','multi_side','final_side'):
            model,opt=self.condition(mode);body=r.old.state_hash(model.body.state_dict());model.train()
            for step in (1,2,3):
                r.old.update(model,opt,self.x,self.y,2200+step,.1)
                row=r.gradient_record(model,step,mode)
                self.assertFalse(row['backbone_gradient_present']);r.validate_optimizer(model,opt,mode,step,self.s)
            self.assertEqual(body,r.old.state_hash(model.body.state_dict()))
            self.assertEqual(sum(len(layer._forward_hooks) for layer in model.body.encoder.layers),0)

    def test_transition_preserves_head_history_and_frozen_weights(self):
        model,opt=self.condition('lp_full');model.train();body=r.old.state_hash(model.body.state_dict())
        for i in range(2):r.old.update(model,opt,self.x,self.y,300+i,.1)
        self.assertEqual(body,r.old.state_hash(model.body.state_dict()));r.validate_optimizer(model,opt,'lp_full',2,self.s)
        history=r.optimizer_hash(opt);r.activate_epoch(model,opt,'lp_full',2,self.s)
        self.assertEqual(history,r.optimizer_hash(opt))
        r.old.update(model,opt,self.x,self.y,302,.1);r.validate_optimizer(model,opt,'lp_full',3,self.s)
        self.assertNotEqual(body,r.old.state_hash(model.body.state_dict()))
        self.assertEqual(int(opt.state[model.head.weight]['step']),3)
        self.assertEqual(int(opt.state[model.body.conv_proj.weight]['step']),1)

    def test_native_full_matches_previous_wrapper(self):
        model,opt=self.condition('full')
        other=copy.deepcopy(self.core);other_opt=r.previous.restore(other,self.cp,self.s,DEVICE,'fresh')
        wrapper=r.t.TokenModel(other,'full',1);model.train();wrapper.train()
        self.assertEqual(r.optimizer_hash(opt),r.optimizer_hash(other_opt))
        for i in range(3):
            for optimizer in (opt,other_opt):optimizer.param_groups[0]['lr']=.0001*(i+1)/3
            loss=r.old.update(model,opt,self.x,self.y,401+i,.1)
            ref=r.old.update(wrapper,other_opt,self.x,self.y,401+i,.1)
            self.assertEqual(loss,ref);self.assertEqual(r.old.state_hash(model.state_dict()),r.old.state_hash(other.state_dict()))
            self.assertEqual(r.optimizer_hash(opt),r.optimizer_hash(other_opt))

    def test_warmup_restoration_and_roundtrip(self):
        for mode in self.s['modes']:
            model,opt=self.condition(mode);initial=copy.deepcopy(model.state_dict());fresh=copy.deepcopy(opt.state_dict())
            model.train();r.old.update(model,opt,self.x,self.y,500,.1)
            model.load_state_dict(initial);opt.load_state_dict(fresh);model.zero_grad(set_to_none=True)
            self.assertEqual(r.old.state_hash(model.state_dict()),r.old.state_hash(initial));self.assertFalse(opt.state)
            r.old.update(model,opt,self.x,self.y,501,.1)
            clone,cloned_opt=self.condition(mode);clone.load_state_dict(model.state_dict());cloned_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
            model.train();clone.train()
            self.assertEqual(r.old.update(model,opt,self.x,self.y,502,.1),r.old.update(clone,cloned_opt,self.x,self.y,502,.1))
            self.assertEqual(r.old.state_hash(model.state_dict()),r.old.state_hash(clone.state_dict()))

    def test_real_shape_counts_and_hook_cleanup(self):
        # Real shape, artificial weights/inputs. Never load a real checkpoint here.
        core=r.m.make_model('full_rgb32',9,False)
        side=r.ResidualSideNet(core,48,(2,4,6),seed=2).to(DEVICE)
        self.assertEqual(sum(p.numel() for p in side.parameters() if p.requires_grad),125002)
        self.assertEqual(sum(p.numel() for p in side.parameters() if not p.requires_grad),2691648)
        side.eval()
        with torch.no_grad():base,features=side.extract(self.x[:1])
        self.assertEqual([tuple(x.shape) for x in features],[(1,65,192)]*3)
        self.assertFalse(any(x.requires_grad for x in features))
        original=side.body.forward
        def fail(*args):raise RuntimeError('artificial hook cleanup test')
        side.body.forward=fail
        try:
            with self.assertRaisesRegex(RuntimeError,'artificial hook'):side.extract(self.x[:1])
        finally:side.body.forward=original
        self.assertEqual(sum(len(layer._forward_hooks) for layer in side.body.encoder.layers),0)

    def test_quality_time_and_launch_gates(self):
        s=json.loads(r.SPEC.read_text());records=[]
        for seed in s['seeds']:
            for mode in s['modes']:
                records.append(dict(seed=seed,mode=mode,initial_accuracy=.82,final_accuracy=.84 if mode in ('full','multi_side') else .83,
                    complete_adaptation_seconds=89. if mode=='multi_side' else 100.))
        self.assertTrue(assess(records,s,False)[1])
        self.assertFalse(assess(records,s,True)[1]);self.assertFalse(assess(records,s,False,False)[1])
        candidate=next(row for row in records if row['mode']=='multi_side');candidate['final_accuracy']=.837
        self.assertTrue(assess(records,s,False)[1]);candidate['final_accuracy']=.8369
        self.assertFalse(assess(records,s,False)[1]);candidate['final_accuracy']=.84
        final=next(row for row in records if row['mode']=='final_side');final.update(final_accuracy=.84,complete_adaptation_seconds=90.)
        self.assertFalse(assess(records,s,False)[1])
        final.update(final_accuracy=.83,complete_adaptation_seconds=100.)
        full=next(row for row in records if row['mode']=='full');full['final_accuracy']=.79
        self.assertFalse(assess(records,s,False)[1])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');assert DEVICE.type=='cpu' or torch.cuda.is_available()
    r.m.configure();unittest.main(argv=['test_stage15_side_adaptation'],verbosity=2)
