"""Native-model gradient, AdamW, split-pool and provenance checks; CPU only."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import stage9_boundary_diagnostic as d

torch.set_num_threads(1)

class Checks(unittest.TestCase):
    def setUp(self):d.v.configure();d.v.seed_all(13011)

    def test_every_native_cut_exact_with_train_dropout(self):
        model=d.make_model(False);nn.init.normal_(model.heads.head.weight,std=.02);model.train()
        x=torch.rand(2,3,32,32);target=.7*F.one_hot(torch.tensor([0,1]),10).float()+.3*F.one_hot(torch.tensor([1,0]),10).float()
        torch.manual_seed(13012);expected=model(x);F.cross_entropy(expected,target,label_smoothing=.1).backward()
        reference={n:p.grad.clone() for n,p in model.named_parameters()}
        counts=[]
        for cut in range(6):
            work=copy.deepcopy(model);work.zero_grad(set_to_none=True);torch.manual_seed(13012)
            logits,h,bridge=d.split_forward(work,x,cut)
            self.assertTrue(torch.equal(logits,expected))
            F.cross_entropy(logits,target,label_smoothing=.1).backward()
            prefix=set(d.named_partition(work,cut));counts.append(sum(p.numel() for n,p in work.named_parameters() if n in prefix))
            for n,p in work.named_parameters():
                if n in prefix:self.assertIsNone(p.grad)
                else:self.assertTrue(torch.equal(p.grad,reference[n]),(cut,n))
            h.backward(bridge.grad)
            for n,p in work.named_parameters():self.assertTrue(torch.equal(p.grad,reference[n]),(cut,n))
        self.assertEqual(counts,sorted(set(counts)))
        self.assertEqual(counts[2],911808)
        self.assertGreater(counts[0],0)

    def test_zero_and_freeze_have_expected_state_and_suffix(self):
        model=d.make_model(True);nn.init.normal_(model.heads.head.weight,std=.02);model.eval()
        opt=d.optimizer(model,.001);x=torch.rand(2,3,32,32);y=torch.tensor([0,1])
        F.cross_entropy(model(x),y).backward();opt.step();model.zero_grad(set_to_none=True)
        weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict())
        F.cross_entropy(model(x),y).backward();grads={n:p.grad.clone() for n,p in model.named_parameters()}
        opt.step();full=copy.deepcopy(model.state_dict())
        for cut in [0,1]:
            names=set(d.named_partition(model,cut))
            for mode in ['zero','freeze']:
                work=d.make_model(True);optimizer=d.optimizer(work,.001);d.restore(work,optimizer,weights,state)
                original=copy.deepcopy(optimizer.state_dict())
                for n,p in work.named_parameters():p.grad=grads[n] if n not in names else (torch.zeros_like(p) if mode=='zero' else None)
                optimizer.step()
                for i,(n,p) in enumerate(work.named_parameters()):
                    if n not in names:self.assertTrue(torch.equal(p,full[n]),n)
                    elif mode=='freeze':
                        self.assertTrue(torch.equal(p,weights[n]),n)
                        self.assertEqual(int(optimizer.state[p]['step']),1)
                        self.assertTrue(torch.equal(optimizer.state[p]['exp_avg'],original['state'][i]['exp_avg']))
                    else:
                        self.assertEqual(int(optimizer.state[p]['step']),2)
                        prior=original['state'][i];beta1,beta2=.9,.999
                        m=beta1*prior['exp_avg'];vv=beta2*prior['exp_avg_sq']
                        expected=weights[n]*(1-.001*.05)-.001*(m/(1-beta1**2))/((vv/(1-beta2**2)).sqrt()+1e-8)
                        torch.testing.assert_close(p,expected,rtol=1e-6,atol=1e-7)

    def test_calibration_and_scoring_pools_disjoint(self):
        y=torch.arange(10).repeat_interleave(20);a,b=d.role_pools(y)
        self.assertFalse(set(a.tolist())&set(b.tolist()))
        self.assertEqual(len(a)+len(b),len(y))
        for pool in [a,b]:self.assertEqual(torch.bincount(y[pool]).tolist(),[10]*10)
        spec=dict(probe_batches_per_checkpoint=4,probe_train_batch_size=2,probe_validation_batch_size=2)
        data=(torch.rand(40,3,32,32),torch.arange(10).repeat_interleave(4),torch.rand(40,3,32,32),torch.arange(10).repeat_interleave(4),{})
        probes=d.prepare_probes(data,1201,5,spec,torch.device('cpu'),False)
        for key in ['training_indices','validation_indices']:
            roles={r:{i for p in probes if p['role']==r for i in p[key]} for r in ['calibration','scoring']}
            self.assertFalse(roles['calibration']&roles['scoring'])
            self.assertEqual(len(roles['calibration']),4);self.assertEqual(len(roles['scoring']),4)

    def test_threshold_never_fits_scoring_outcomes(self):
        cal=[dict(x=x) for x in [0.,1.,2.,3.]]
        threshold,a=d.threshold_selection(cal,[dict(x=x) for x in [0.,10.,20.]],'x',.4)
        other,b=d.threshold_selection(cal,[dict(x=x) for x in [-100.,1000.,2000.]],'x',.4)
        self.assertEqual(threshold,other);self.assertEqual(len(a),1);self.assertEqual(len(b),1)
        self.assertAlmostEqual(d.spearman([1,2,3],[6,5,4]),-1)
        self.assertEqual(d.ranked([1,1,3]),[.5,.5,2.])

    def test_completion_hash_and_incomplete_attempts(self):
        with tempfile.TemporaryDirectory(prefix='stage9_unit_') as tmp:
            parent=Path(tmp);first=d.next_attempt(parent);second=d.next_attempt(parent)
            self.assertNotEqual(first,second);self.assertTrue(first.exists())
            path=first/'evidence.txt';path.write_text('original')
            d.v.atomic_json(parent/'COMPLETE.json',dict(fingerprint='abc',sha256={str(path):d.sha(path)}))
            self.assertIsNotNone(d.validate_complete(parent,'abc'))
            with self.assertRaises(AssertionError):d.validate_complete(parent,'other')
            path.write_text('changed')
            with self.assertRaises(AssertionError):d.validate_complete(parent,'abc')

if __name__=='__main__':unittest.main(verbosity=2)
