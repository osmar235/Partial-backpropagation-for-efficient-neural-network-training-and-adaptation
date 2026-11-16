"""Controls for prefix omission, qualified baseline parity and prospective decisions."""
import argparse
import copy
from collections import Counter
import unittest
from pathlib import Path
import torch
import prefix_adaptation as c
from review_prefix_adaptation import decision,refresh_assessment,quality_deficits,verify_saved_result
DEVICE=torch.device('cpu')


class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=c.configuration(True,DEVICE);self.spec=self.cfg['spec'];self.seed=self.spec['seeds'][0]
        x,y,_,_,_=c.r.v.load_data('development',DEVICE,fake=True)
        self.x=c.r.v.frozen6.normalize_cifar(x[:3]);self.y=y[:3]

    def test_order_balance_and_global_cadence(self):
        spec=c.read(c.SPEC)
        for seed in spec['seeds']:
            orders=[tuple(a for s,a in c.schedule(spec,i) if s==seed) for i in range(6)]
            self.assertEqual(len(set(orders)),6)
            for arm in spec['arms']:self.assertEqual(Counter(o.index(arm) for o in orders),{0:2,1:2,2:2})
        self.assertEqual([i for i in range(12) if c.full_update('periodic_cut3_r4',i,self.spec)],[0,4,8])
        self.assertEqual(sum(c.full_update('periodic_cut3_r4',i,spec) for i in range(880)),220)
        self.assertEqual(sum(c.full_update('frozen_cut3',i,spec) for i in range(880)),0)

    def test_full_matches_qualified_implementation(self):
        oldcfg=c.historical.configuration(True,DEVICE);policy=self.spec['execution_policy']
        reference,ropt,_=c.historical.make(self.seed,policy,oldcfg,DEVICE)
        model,opt,_=c.make(self.seed,'full',self.cfg,DEVICE);router=c.router_for(model,'full',self.spec)
        reference.train();model.train()
        with c.e.policy(policy):
            for i in range(9):
                a=c.historical.update(reference,ropt,self.x,self.y,197001+i,policy,oldcfg['spec'])
                b=c.update(model,opt,self.x,self.y,197001+i,'full',self.spec,router,i)
                self.assertEqual(a,b)
        self.assertEqual(c.r.old.state_hash(reference.state_dict()),c.r.old.state_hash(model.state_dict()))
        self.assertEqual(c.r.optimizer_hash(ropt),c.r.optimizer_hash(opt))

    def test_all_arm_ages_frozen_prefix_and_stale_gradients(self):
        for arm in self.spec['arms']:
            model,opt,cp=c.make(self.seed,arm,self.cfg,DEVICE);model.train();router=c.router_for(model,arm,self.spec)
            after_full=None;moments=None
            with c.e.policy(self.spec['execution_policy']):
                for i in range(12):
                    c.update(model,opt,self.x,self.y,198001+i,arm,self.spec,router,i)
                    if arm=='periodic_cut3_r4' and i==0:
                        after_full={n:p.detach().clone() for n,p in router.prefix.items()}
                        moments={n:copy.deepcopy(opt.state[p]) for n,p in router.prefix.items()}
                    if arm=='periodic_cut3_r4' and i in (1,2,3):
                        for n,p in router.prefix.items():
                            self.assertIsNone(p.grad);self.assertTrue(torch.equal(p,after_full[n]))
                            for key,value in moments[n].items():self.assertTrue(torch.equal(opt.state[p][key],value))
            result=c.validate_state(model,opt,arm,12,self.spec,cp['model_state'])
            self.assertEqual(result['prefix_updates'],{'full':12,'frozen_cut3':0,'periodic_cut3_r4':3}[arm])
            self.assertEqual(result['suffix_updates'],12)
            # A corrupt counter must fail the auditor's independent loaded-state check.
            active=next(p for p in model.parameters() if p in opt.state)
            opt.state[active]['step'].add_(1)
            with self.assertRaises(AssertionError):c.validate_state(model,opt,arm,12,self.spec,cp['model_state'],False)

    def test_warmup_restores_empty_moments_flags_and_weights(self):
        for arm in self.spec['arms']:
            model,opt,cp=c.make(self.seed,arm,self.cfg,DEVICE);initial=copy.deepcopy(opt.state_dict())
            model.train();router=c.router_for(model,arm,self.spec)
            with c.e.policy(self.spec['execution_policy']):
                for i in range(3):c.update(model,opt,self.x,self.y,199001+i,arm,self.spec,router,i)
            model.load_state_dict(cp['model_state']);opt.load_state_dict(initial);model.zero_grad(set_to_none=True)
            model.requires_grad_(True);router=c.router_for(model,arm,self.spec)
            self.assertFalse(opt.state);self.assertTrue(router.active)
            self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertEqual(c.r.old.state_hash(model.state_dict()),c.r.old.state_hash(cp['model_state']))

    def test_recipe_and_quality_modes(self):
        spec=c.read(c.SPEC);old=c.read(c.historical.SPEC)
        for key in ('seeds','epochs','batch_size','eval_batch_size','expected_batches_per_epoch','source_optimizer_age',
                    'learning_rate','lr_warmup_epochs','min_lr_ratio','weight_decay','mixup_alpha','label_smoothing',
                    'batch_seed_offset','augmentation_seed_offset','mixup_seed_offset','permutation_seed_offset','dropout_seed_offset'):
            self.assertEqual(spec[key],old[key])
        ref=dict(final_accuracy=.85,deployed_accuracy=.84)
        candidate=dict(final_accuracy=.848,deployed_accuracy=.838)
        a,b,ok=quality_deficits(ref,candidate,spec);self.assertTrue(ok);self.assertAlmostEqual(a,.2);self.assertAlmostEqual(b,.2)
        candidate['deployed_accuracy']=.836;self.assertFalse(quality_deficits(ref,candidate,spec)[2])

    def test_joint_gate_and_refresh_interpretation(self):
        spec=c.read(c.SPEC)
        rows=[dict(arm=arm,seed=seed,session=s,saving_percent=5.,baseline_ready=True,quality_pass=True,
                   common_deficit_pp=.1,deployed_deficit_pp=.1,candidate_accuracy=.84,deployed_accuracy=.84,candidate_seconds=5.)
              for arm in spec['arms'][1:] for seed in spec['seeds'] for s in range(6)]
        good=decision(rows,spec,[],False);self.assertTrue(all(r['practical_screen_pass'] for r in good))
        self.assertEqual(good[0]['adjusted_both_source_direction_p'],.03125)
        both=refresh_assessment(rows,good,spec,[],False);self.assertFalse(both['refresh_needed_for_fixed_screen'])
        for r in rows:
            if r['arm']=='frozen_cut3':r['quality_pass']=False
        changed=decision(rows,spec,[],False)
        self.assertTrue(refresh_assessment(rows,changed,spec,[],False)['refresh_needed_for_fixed_screen'])
        rows[0]['quality_pass']=True
        self.assertFalse(refresh_assessment(rows,decision(rows,spec,[],False),spec,[],False)['refresh_needed_for_fixed_screen'])
        for field,value in [('baseline_ready',False),('quality_pass',False),('saving_percent',0.)]:
            failed=copy.deepcopy(rows);next(r for r in failed if r['arm']=='periodic_cut3_r4')[field]=value
            self.assertFalse(decision(failed,spec,[],False)[1]['practical_screen_pass'])
        self.assertFalse(any(r['practical_screen_pass'] for r in decision(rows,spec,[0],False)))
        self.assertFalse(any(r['practical_screen_pass'] for r in decision(rows,spec,[],True)))

    def test_restart_rescore_semantics(self):
        saved=dict(status='PASS',conditions=[False],rescores={'x':dict(max_abs=0.)})
        current=copy.deepcopy(saved);current['rescores']['x']['max_abs']=1e-7
        verify_saved_result(saved,current)
        current['conditions']=[True]
        with self.assertRaises(AssertionError):verify_saved_result(saved,current)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device(args.device);c.r.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        c.r.v.atomic_json(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
