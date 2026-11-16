"""Check matched learning/inference, timing gates and independent stream evidence."""
import argparse
import copy
import unittest
import numpy as np
import torch
import qkv_training as e
from review_qkv_training import qualified,paired_identity

DEVICE=torch.device('cpu')

def small_cfg():
    spec=e.read(e.SPEC)
    spec.update(seeds=[9971,9972],sessions=2,epochs=2,batch_size=3,eval_batch_size=4,
        expected_batches_per_epoch=6,source_optimizer_age=2,hardware_warmup_steps=1)
    return dict(spec=spec,smoke=True,checkpoints={})

class Tests(unittest.TestCase):
    def test_order_balance(self):
        spec=e.read(e.SPEC)
        for seed in spec['seeds']:
            native_first=[];positions=[]
            for session in range(8):
                order=e.schedule(spec,session)
                native_first.append(order.index((seed,'native'))<order.index((seed,'qkv_unbind')))
                positions.append(min(order.index((seed,a)) for a in spec['arms']))
            self.assertEqual(sum(native_first),4)
            self.assertEqual(positions.count(0),4);self.assertEqual(positions.count(2),4)

    def test_identity_and_quality_override_speed(self):
        values=[5.]*8
        yes=qualified(values,True,True,True,False)
        self.assertTrue(yes['confirmed']);self.assertEqual(yes['statistics']['bonferroni_p'],.0078125)
        for flags in [(False,True,True,False),(True,False,True,False),(True,True,False,False),(True,True,True,True)]:
            self.assertFalse(qualified(values,*flags)['confirmed'])
        self.assertFalse(qualified([5.]*7+[0.],True,True,True,False)['confirmed'])

    def test_pair_rejects_trajectory_changes(self):
        a=dict(summary={k:'same' for k in ('initial_model_sha256','initial_optimizer_sha256','final_model_sha256','final_optimizer_sha256','final_global_rng')},
               steps=[{'loss':'1.0'}],stream_hash='s',predictions={'labels':np.array([0]),'epoch_0':np.array([[1.,2.]])})
        self.assertTrue(paired_identity(a,copy.deepcopy(a)))
        for k in a['summary']:
            b=copy.deepcopy(a);b['summary'][k]='changed';self.assertFalse(paired_identity(a,b))
        b=copy.deepcopy(a);b['predictions']['epoch_0'][0,0]+=1;self.assertFalse(paired_identity(a,b))
        b=copy.deepcopy(a);b['steps'][0]['loss']='1.1';self.assertFalse(paired_identity(a,b))

    def test_eval_and_augmented_training_identity(self):
        cfg=small_cfg();spec=cfg['spec'];seed=9971
        x,y,ev,ey,_=e.r.v.load_data('development',DEVICE,fake=True)
        trajectories=[]
        for arm in spec['arms']:
            model,opt,cp=e.make(seed,arm,cfg,DEVICE)
            self.assertFalse(opt.state)
            before=e.r.old.state_hash(model.state_dict())
            self.assertEqual(before,e.r.old.state_hash(cp['model_state']))
            with e.q.sdpa_kernel(e.q.SDPBackend.MATH):
                initial=e.r.t.predictions(model,ev,4)
                bg,ag,mg,pg=e.r.previous.generators(seed,spec);losses=[];model.train()
                for step,indices in enumerate(e.r.v.frozen6.epoch_batches(len(y),3,bg)):
                    ix=indices.to(DEVICE);xx=e.r.v.frozen6.augment_and_normalize(x[ix],ag)
                    xx,targets,_=e.r.v.mix_batch(xx,y[ix],mg,pg,spec['mixup_alpha'],10)
                    losses.append(e.r.old.update(model,opt,xx,targets,seed+spec['dropout_seed_offset']+step,spec['label_smoothing']))
                final=e.r.t.predictions(model,ev,4)
            e.r.validate_optimizer(model,opt,'full',6,spec)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            trajectories.append(dict(initial=initial,final=final,losses=losses,model=e.r.old.state_hash(model.state_dict()),opt=e.r.optimizer_hash(opt)))
        a,b=trajectories
        self.assertTrue(np.array_equal(a['initial'],b['initial']));self.assertTrue(np.array_equal(a['final'],b['final']))
        for k in ('losses','model','opt'):self.assertEqual(a[k],b[k])

    def test_recipe_retains_reference(self):
        spec=e.read(e.SPEC);ref=e.read(e.r.SPEC)
        for k in ('seeds','epochs','batch_size','eval_batch_size','expected_batches_per_epoch','source_optimizer_age',
                  'learning_rate','lr_warmup_epochs','min_lr_ratio','weight_decay','mixup_alpha','label_smoothing',
                  'batch_seed_offset','augmentation_seed_offset','mixup_seed_offset','permutation_seed_offset','dropout_seed_offset'):
            self.assertEqual(spec[k],ref[k])
        self.assertEqual(spec['sessions']*len(spec['seeds'])*len(spec['arms'])*spec['epochs']*spec['expected_batches_per_epoch'],28160)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');e.r.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
