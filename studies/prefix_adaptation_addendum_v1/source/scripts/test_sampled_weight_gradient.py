"""Artificial correctness checks; never load real checkpoints or datasets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import itertools
import json
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
import stage10_models as m
import stage10_representation as old
from sampled_weight_gradient import WeightGradientFunction,install,set_sampling_seed

DEVICE=torch.device('cpu');DIAGNOSTICS={}

class SamplingTests(unittest.TestCase):
    def model(self,kept=None):
        core=m.make_model('full_rgb32',9871,False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(123)
            with torch.no_grad():core.head.weight.normal_(std=.02)
        if kept is not None:install(core,kept)
        return core.to(DEVICE)

    def data(self):
        gen=torch.Generator().manual_seed(492)
        return torch.randn(3,3,32,32,generator=gen).to(DEVICE),torch.tensor([0,1,2],device=DEVICE)

    def test_exact_enumerated_expectation(self):
        # Enumerate all 2-of-4 subsets; this is a deterministic unbiasedness test.
        gen=torch.Generator().manual_seed(992)
        x=torch.randn(2,5,3,generator=gen,dtype=torch.double).to(DEVICE).requires_grad_()
        w=torch.randn(4,3,generator=gen,dtype=torch.double).to(DEVICE).requires_grad_()
        b=torch.randn(4,generator=gen,dtype=torch.double).to(DEVICE).requires_grad_()
        g=torch.randn(2,5,4,generator=gen,dtype=torch.double).to(DEVICE)
        exact=torch.autograd.grad(F.linear(x,w,b),(x,w,b),g)
        estimates=[]
        for subset in itertools.combinations(range(1,5),2):
            positions=torch.tensor([0,*subset],device=DEVICE)
            indices=(torch.arange(2,device=DEVICE)[:,None]*5+positions).reshape(-1)
            factors=torch.tensor([1.,2.,2.],device=DEVICE,dtype=torch.double).repeat(2)
            z=WeightGradientFunction.apply(x,w,b,indices,factors)
            actual=torch.autograd.grad(z,(x,w,b),g)
            torch.testing.assert_close(actual[0],exact[0],rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(actual[2],exact[2],rtol=1e-12,atol=1e-12);estimates.append(actual[1])
        torch.testing.assert_close(torch.stack(estimates).mean(0),exact[1],rtol=1e-12,atol=1e-12)

    def test_full_rate_gradcheck(self):
        gen=torch.Generator().manual_seed(882)
        x=torch.randn(2,3,2,generator=gen,dtype=torch.double).to(DEVICE).requires_grad_()
        w=torch.randn(3,2,generator=gen,dtype=torch.double).to(DEVICE).requires_grad_()
        b=torch.randn(3,generator=gen,dtype=torch.double).to(DEVICE).requires_grad_()
        indices=torch.arange(6,device=DEVICE);factors=torch.ones(6,device=DEVICE,dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda a,c,d:WeightGradientFunction.apply(a,c,d,indices,factors),(x,w,b)))

    def test_native_forward_and_rng(self):
        native=self.model();sampled=self.model(16);x,y=self.data()
        self.assertEqual(list(native.state_dict()),list(sampled.state_dict()))
        self.assertEqual(old.state_hash(native.state_dict()),old.state_hash(sampled.state_dict()))
        for train in (True,False):
            native.train(train);sampled.train(train)
            torch.manual_seed(912);z=native(x);cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state() if DEVICE.type=='cuda' else None
            torch.manual_seed(912);set_sampling_seed(sampled,42);q=sampled(x)
            torch.testing.assert_close(q,z,atol=0,rtol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(),cpu_rng))
            if gpu_rng is not None:self.assertTrue(torch.equal(torch.cuda.get_rng_state(),gpu_rng))
        DIAGNOSTICS['forward_max_abs_error']=0.

    def gradients(self,kept,seed=46):
        model=self.model(kept);x,y=self.data();x.requires_grad_();model.train();torch.manual_seed(916)
        set_sampling_seed(model,seed);F.cross_entropy(model(x),y).backward()
        return model,{name:p.grad.detach().clone() for name,p in model.named_parameters()},x.grad

    def test_exact_input_and_unsampled_gradients(self):
        native,full,dx=self.gradients(None);sampled,approx,sx=self.gradients(16)
        torch.testing.assert_close(sx,dx,rtol=1e-4,atol=2e-7)
        sampled_names={name for name in full if '.mlp.' in name and name.endswith('weight')}
        for name in full:
            self.assertTrue(bool(torch.isfinite(approx[name]).all()))
            if name not in sampled_names:torch.testing.assert_close(approx[name],full[name],rtol=1e-4,atol=2e-6)
        self.assertTrue(any(not torch.equal(full[n],approx[n]) for n in sampled_names))
        a=torch.cat([approx[n].flatten() for n in sampled_names]);b=torch.cat([full[n].flatten() for n in sampled_names])
        DIAGNOSTICS.update(input_gradient_max_abs_error=float((sx-dx).abs().max()),
            sampled_mlp_weight_gradient_cosine=float(F.cosine_similarity(a,b,dim=0)),
            sampled_mlp_weight_gradient_relative_error=float((a-b).norm()/b.norm()))

    def test_full_rate_network_gradients(self):
        native,full,dx=self.gradients(None);exact,custom,sx=self.gradients(64)
        for name in full:torch.testing.assert_close(custom[name],full[name],rtol=1e-4,atol=2e-6)
        torch.testing.assert_close(sx,dx,rtol=1e-4,atol=2e-7)
        DIAGNOSTICS['full_rate_max_abs_parameter_gradient_error']=max(float((custom[n]-full[n]).abs().max()) for n in full)

    def test_seed_replay_and_all_layers_learn(self):
        model,a,_=self.gradients(16,701);_,b,_=self.gradients(16,701);_,c,_=self.gradients(16,702)
        for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(a[n],c[n]) for n in a if '.mlp.' in n and n.endswith('weight')))
        before=copy.deepcopy(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.05)
        opt.step()
        self.assertTrue(all(int(opt.state[p]['step'])==1 for p in model.parameters()))
        for index,block in enumerate(model.body.encoder.layers):
            for key in ('mlp.0.weight','mlp.3.weight','self_attention.in_proj_weight'):
                name=f'body.encoder.layers.encoder_layer_{index}.{key}'
                self.assertFalse(torch.equal(before[name],model.state_dict()[name]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',device=str(DEVICE),
            tests=result.testsRun,artificial_only=True,diagnostics=DIAGNOSTICS),indent=2)+'\n',encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
