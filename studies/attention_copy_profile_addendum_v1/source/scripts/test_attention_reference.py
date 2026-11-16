"""Small deterministic correctness and benchmark-design checks."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import torch
from torch import nn
import attention_reference as e
from attention_reference_adapter import ReferenceSelfAttention

DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def test_outputs_all_derivatives_and_parameter_mapping(self):
        torch.manual_seed(99174)
        native=nn.MultiheadAttention(12,3,dropout=0,batch_first=True).to(DEVICE).double()
        source=copy.deepcopy(native);optimizer=torch.optim.AdamW(source.parameters(),lr=1e-4)
        names={n:id(p) for n,p in source.named_parameters()};before=e.q.rng(DEVICE)
        changed=ReferenceSelfAttention(source)
        self.assertEqual(names,{n:id(p) for n,p in changed.named_parameters()})
        self.assertEqual(before,e.q.rng(DEVICE));self.assertEqual(set(native.state_dict()),set(changed.state_dict()))
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},set(names.values()))
        x=torch.randn(2,7,12,device=DEVICE,dtype=torch.float64,requires_grad=True)
        xx=x.detach().clone().requires_grad_();up=torch.randn_like(x)
        with e.q.sdpa_kernel(e.q.SDPBackend.MATH):
            a=native(x,x,x,need_weights=False)[0];b=changed(xx,xx,xx,need_weights=False)[0]
            self.assertEqual(a.stride(),b.stride())
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12);a.backward(up);b.backward(up)
        torch.manual_seed(99175);da=torch.nn.functional.dropout(a,.1,True)
        torch.manual_seed(99175);db=torch.nn.functional.dropout(b,.1,True)
        torch.testing.assert_close(da,db,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(x.grad,xx.grad,rtol=1e-12,atol=1e-12)
        for n,p in native.named_parameters():torch.testing.assert_close(p.grad,dict(changed.named_parameters())[n].grad,rtol=1e-12,atol=1e-12)

    def test_schedule_balance_and_exclusions(self):
        spec=e.read(e.SPEC)
        for session in range(4):
            orders=[e.order(spec,session,2,ri) for ri in range(6)]
            self.assertEqual(len({tuple(o) for o in orders}),6)
            for arm in spec['arms']:
                self.assertEqual([sum(o[pos]==arm for o in orders) for pos in range(3)],[2,2,2])

    def test_full_model_snapshots_and_sensitive_gate(self):
        cfg=e.configuration(True,DEVICE);spec=cfg['spec'];x,y=e.inputs(99173,3,spec,DEVICE)
        initial,ref=e.snapshots(99173,'math',cfg,x,y,DEVICE)
        for arm in spec['arms']:
            other,actual=e.snapshots(99173,arm,cfg,x,y,DEVICE)
            self.assertEqual(initial,other)
            # Pipeline accepts and records a genuine reference gate failure.
            comparison=e.q.compare(ref,actual,spec['tolerances'])
            self.assertIn(comparison['status'],('PASS','FAIL_NUMERICAL'))
            self.assertTrue(comparison['rng_match'])
            # Initial full-model outputs must agree, including downstream
            # dropout. Later Adam drift can still legitimately fail its gate.
            torch.testing.assert_close(ref[0]['logits'],actual[0]['logits'],**spec['tolerances']['logits'])
            if arm!='timm_reference':self.assertEqual(comparison['status'],'PASS')
        broken=copy.deepcopy(ref)
        key=next(n for n in broken[0]['parameter'] if n.endswith('in_proj_bias'))
        broken[0]['parameter'][key][0]+=1
        self.assertEqual(e.q.compare(ref,broken,spec['tolerances'])['status'],'FAIL_NUMERICAL')

    def test_unsupported_inputs(self):
        module=ReferenceSelfAttention(nn.MultiheadAttention(12,3,batch_first=True).to(DEVICE))
        x=torch.ones(2,7,12,device=DEVICE)
        with self.assertRaises(AssertionError):module(x,x.clone(),x,need_weights=False)
        with self.assertRaises(AssertionError):module(x,x,x,need_weights=True)
        with self.assertRaises(AssertionError):module(x,x,x,need_weights=False,is_causal=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path)
    args=ap.parse_args();DEVICE=torch.device(args.device);e.q.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        e.save(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
