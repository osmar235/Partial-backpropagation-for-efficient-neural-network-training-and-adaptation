"""Check exact parameters/state mapping, all derivatives, and the split mechanism."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.func import functional_call
import qkv_layout as e
from qkv_unbind import UnbindSelfAttention,install
DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def test_attention_outputs_and_all_gradients(self):
        torch.manual_seed(99151)
        native=nn.MultiheadAttention(12,3,dropout=0,batch_first=True).to(device=DEVICE,dtype=torch.float64)
        changed=UnbindSelfAttention(copy.deepcopy(native))
        x=torch.randn(2,7,12,device=DEVICE,dtype=torch.float64,requires_grad=True)
        xx=x.detach().clone().requires_grad_();upstream=torch.randn_like(x)
        with e.sdpa_kernel(e.SDPBackend.MATH):
            a=native(x,x,x,need_weights=False)[0];b=changed(xx,xx,xx,need_weights=False)[0]
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            a.backward(upstream);b.backward(upstream)
        torch.testing.assert_close(x.grad,xx.grad,rtol=1e-12,atol=1e-12)
        for (name,p),(other,q) in zip(native.named_parameters(),changed.named_parameters()):
            self.assertEqual(name,other);torch.testing.assert_close(p.grad,q.grad,rtol=1e-12,atol=1e-12)

    def test_double_precision_finite_difference(self):
        torch.manual_seed(99152)
        module=UnbindSelfAttention(nn.MultiheadAttention(6,3,dropout=0,batch_first=True).double().to(DEVICE))
        names=list(dict(module.named_parameters()));parameters=tuple(module.parameters())
        x=torch.randn(1,3,6,device=DEVICE,dtype=torch.float64,requires_grad=True)
        def fn(x,*values):return functional_call(module,dict(zip(names,values)),(x,x,x),{'need_weights':False})[0]
        with e.sdpa_kernel(e.SDPBackend.MATH):
            self.assertTrue(torch.autograd.gradcheck(fn,(x,*parameters),eps=1e-6,atol=1e-5,rtol=1e-3,fast_mode=True))

    def test_existing_optimizer_and_checkpoint_mapping(self):
        spec=json.loads(e.SPEC.read_text());model=e.make(99153,spec,True).to(DEVICE)
        opt=e.optimizer(model,spec);x=torch.ones(2,3,32,32,device=DEVICE);y=torch.tensor([1,2],device=DEVICE)
        with e.sdpa_kernel(e.SDPBackend.MATH):e.update(model,opt,x,y,99153,0,spec)
        before_state=e.old.state_hash(model.state_dict());before_opt=e.opt_hash(opt)
        names={n:id(p) for n,p in model.named_parameters()};random=e.rng(DEVICE)
        install(model)
        self.assertEqual(e.rng(DEVICE),random);self.assertEqual(names,{n:id(p) for n,p in model.named_parameters()})
        self.assertEqual(before_state,e.old.state_hash(model.state_dict()));self.assertEqual(before_opt,e.opt_hash(opt))
        original=e.make(99153,spec,True).to(DEVICE);original.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(e.old.state_hash(original.state_dict()),before_state)
        with e.sdpa_kernel(e.SDPBackend.MATH):e.update(model,opt,x,y,99153,1,spec)
        self.assertTrue(all(int(opt.state[p]['step'])==2 for p in model.parameters()))

    def test_full_model_two_updates_and_rejection(self):
        spec=json.loads(e.SPEC.read_text());x=torch.ones(3,3,32,32,device=DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        a=e.snapshots(99154,'math',spec,True,x,y,DEVICE);b=e.snapshots(99154,'qkv_unbind',spec,True,x,y,DEVICE)
        self.assertEqual(e.compare(a,b,spec['tolerances'])['status'],'PASS')
        b[0]['gradient'][next(iter(b[0]['gradient']))].flatten()[0]+=1
        self.assertEqual(e.compare(a,b,spec['tolerances'])['status'],'FAIL_NUMERICAL')

    def test_mechanism_and_unsupported_inputs(self):
        spec=json.loads(e.SPEC.read_text());x=torch.ones(3,3,32,32,device=DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        probes={}
        for arm in ('math','qkv_unbind'):
            rec=e.probe(99155,3,arm,spec,True,x,y,DEVICE)
            probes[arm]=rec
            self.assertEqual(rec['status'],'SUPPORTED');self.assertEqual(rec['unbind_backward_nodes'],2 if arm=='qkv_unbind' else 0)
        self.assertEqual(probes['math']['select_backward_nodes']-probes['qkv_unbind']['select_backward_nodes'],6)
        module=UnbindSelfAttention(nn.MultiheadAttention(12,3,batch_first=True)).to(DEVICE)
        x=torch.ones(2,3,12,device=DEVICE)
        with self.assertRaises(AssertionError):module(x,x.clone(),x,need_weights=False)
        with self.assertRaises(AssertionError):module(x,x,x,need_weights=True)
        with self.assertRaises(AssertionError):module(x,x,x,need_weights=False,is_causal=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');e.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',device=str(DEVICE),tests=result.testsRun),indent=2)+'\n',encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
