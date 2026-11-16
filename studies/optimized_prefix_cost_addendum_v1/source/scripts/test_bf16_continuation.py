"""Meaningful controls for precision scoring, state reset and prospective gates."""
import argparse
import copy
from collections import Counter
from pathlib import Path
import unittest
import numpy as np
import torch
import bf16_continuation as c
from review_bf16_continuation import decision,verify_saved_result

DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def test_order_balance(self):
        spec=c.read(c.SPEC)
        for seed in spec['seeds']:
            orders=[];locations=[]
            for session in range(spec['sessions']):
                schedule=c.schedule(spec,session);orders.append(tuple(a for s,a in schedule if s==seed));locations.append(min(i for i,(s,a) in enumerate(schedule) if s==seed))
            self.assertEqual(len(set(orders)),6);self.assertEqual(locations.count(0),3)
            for arm in spec['arms']:self.assertEqual(Counter(o.index(arm) for o in orders),{0:2,1:2,2:2})

    def test_quality_baseline_fragmentation_override_timing(self):
        spec=c.read(c.SPEC)
        rows=[dict(arm=arm,seed=seed,session=s,saving_percent=5.,baseline_ready=True,quality_pass=True,common_deficit_pp=.1,deployed_deficit_pp=.2)
              for arm in spec['arms'][1:] for seed in spec['seeds'] for s in range(6)]
        yes=decision(rows,spec,[],False);self.assertTrue(all(r['practical_screen_pass'] for r in yes));self.assertEqual(yes[0]['adjusted_both_source_direction_p'],.03125)
        for field,value in [('quality_pass',False),('baseline_ready',False),('saving_percent',0.)]:
            changed=copy.deepcopy(rows);changed[0][field]=value
            self.assertFalse(decision(changed,spec,[],False)[0]['practical_screen_pass'])
        self.assertFalse(decision(rows,spec,[0],False)[0]['practical_screen_pass']);self.assertFalse(decision(rows,spec,[],True)[0]['practical_screen_pass'])

    def test_explicit_fp32_loss_and_gradients(self):
        cfg=c.configuration(True,DEVICE);spec=cfg['spec'];x,y,_,_,_=c.r.v.load_data('development',DEVICE,fake=True)
        model,_,_=c.make(spec['seeds'][0],spec['arms'][0],cfg,DEVICE);model.eval();x=c.r.v.frozen6.normalize_cifar(x[:3])
        for arm in spec['arms']:
            for targets in (y[:3],torch.nn.functional.one_hot(y[:3],10).float()*.8+.02):
                with c.e.policy(arm):
                    model.zero_grad(set_to_none=True);logits,loss=c.forward_loss(model,x,targets,arm,spec)
                    expected=c.F.cross_entropy(logits.float(),targets,label_smoothing=spec['label_smoothing'])
                    self.assertTrue(torch.equal(loss,expected));self.assertEqual(loss.dtype,torch.float32)
                    loss.backward();self.assertTrue(all(p.grad is not None and p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_common_initial_state_warmup_restore_and_own_eval(self):
        cfg=c.configuration(True,DEVICE);spec=cfg['spec'];x,y,ev,ey,_=c.r.v.load_data('development',DEVICE,fake=True);baseline=None
        for arm in spec['arms']:
            model,opt,cp=c.make(spec['seeds'][0],arm,cfg,DEVICE);before=c.r.old.state_hash(model.state_dict());state=copy.deepcopy(opt.state_dict())
            self.assertFalse(opt.state);self.assertEqual(before,c.r.old.state_hash(cp['model_state']))
            common=c.predictions(model,ev,4,'reference_fp32_math')
            if baseline is not None:self.assertTrue(np.array_equal(common,baseline))
            baseline=common;model.train()
            with c.e.policy(arm):c.update(model,opt,c.r.v.frozen6.normalize_cifar(x[:3]),y[:3],199,arm,spec)
            self.assertNotEqual(before,c.r.old.state_hash(model.state_dict()))
            c.r.validate_optimizer(model,opt,'full',1,spec)
            model.load_state_dict(cp['model_state']);opt.load_state_dict(state);self.assertFalse(opt.state)
            self.assertEqual(before,c.r.old.state_hash(model.state_dict()))
            self.assertEqual(c.predictions(model,ev,4,arm).shape,(8,10))

    def test_fixed_recipe_and_budget(self):
        spec=c.read(c.SPEC);old=c.read(c.b.SPEC)
        for key in ('seeds','epochs','batch_size','eval_batch_size','expected_batches_per_epoch','source_optimizer_age','learning_rate','lr_warmup_epochs','min_lr_ratio','weight_decay','mixup_alpha','label_smoothing','batch_seed_offset','augmentation_seed_offset','mixup_seed_offset','permutation_seed_offset','dropout_seed_offset'):
            self.assertEqual(spec[key],old[key])
        self.assertEqual(spec['sessions']*len(spec['seeds'])*len(spec['arms'])*spec['epochs']*spec['expected_batches_per_epoch'],31680)

    def test_restart_preserves_science_despite_valid_rescore_variation(self):
        a=dict(status='PASS',conditions=[dict(practical_screen_pass=False)],rescores={'case':{'max_abs':0.}})
        b=copy.deepcopy(a);b['rescores']['case']['max_abs']=1e-7
        verify_saved_result(a,b)
        b['conditions'][0]['practical_screen_pass']=True
        with self.assertRaises(AssertionError):verify_saved_result(a,b)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device(args.device);c.r.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists()
        c.r.v.atomic_json(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
