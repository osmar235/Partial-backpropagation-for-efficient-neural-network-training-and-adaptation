"""Meaningful gate, dtype, stochastic-mode and policy checks on tiny inputs."""
import argparse
import copy
from pathlib import Path
import unittest
import torch
from torch import nn
import optimized_execution as e

DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def test_precision_budgets_detect_errors(self):
        cfg=e.configuration(True,DEVICE);spec=cfg['spec'];x,y=e.prior.inputs(99203,3,spec,DEVICE)
        _,reference=e.snapshots(99203,'reference_fp32_math',cfg,x,y,DEVICE)
        self.assertEqual(e.compare(reference,reference,'reference_bf16_math',spec)['status'],'PASS_SANITY_ONLY')
        corrupted=copy.deepcopy(reference);corrupted[0]['logits']+=.1
        self.assertEqual(e.compare(reference,corrupted,'reference_bf16_math',spec)['status'],'FAIL_NUMERICAL_SANITY')
        corrupted=copy.deepcopy(reference)
        for key in corrupted[0]['gradient']:corrupted[0]['gradient'][key].neg_()
        result=e.compare(reference,corrupted,'reference_bf16_math',spec)
        self.assertTrue(any(c['budget']=='gradient_cosine_min' and not c['pass_budget'] for c in result['checks']))
        corrupted=copy.deepcopy(reference);name=next(n for n in corrupted[0]['parameter'] if n.endswith('in_proj_bias'))
        corrupted[0]['parameter'][name][0]+=1
        self.assertEqual(e.compare(reference,corrupted,'native_fp32_math',spec)['status'],'FAIL_NUMERICAL')

    def test_policy_restored_and_position_balance(self):
        spec=e.read(e.SPEC);e.q.m.configure()
        with e.policy('reference_bf16_auto'):
            self.assertFalse(torch.are_deterministic_algorithms_enabled());self.assertTrue(torch.backends.cuda.flash_sdp_enabled())
        self.assertTrue(torch.are_deterministic_algorithms_enabled());self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        for session in range(4):
            orders=[e.order(spec,session,1,i) for i in range(6)]
            for arm in spec['arms']:
                self.assertEqual([sum(row[pos]==arm for row in orders) for pos in range(6)],[1]*6)

    def test_dropout_disabled_only_in_qualification(self):
        cfg=e.configuration(True,DEVICE)
        model=e.make(99203,'reference_fp32_math',cfg,DEVICE)
        qualification=e.make(99203,'reference_fp32_math',cfg,DEVICE,qualification=True)
        self.assertTrue(any(m.p>0 for m in model.modules() if isinstance(m,nn.Dropout)))
        self.assertTrue(all(m.p==0 for m in qualification.modules() if isinstance(m,nn.Dropout)))
        self.assertEqual(e.q.old.state_hash(model.state_dict()),e.q.old.state_hash(qualification.state_dict()))

    def test_real_autocast_and_all_parameter_gradients(self):
        if DEVICE.type!='cuda':self.skipTest('CUDA BF16 screen')
        cfg=e.configuration(True,DEVICE);spec=cfg['spec'];x,y=e.prior.inputs(99203,3,spec,DEVICE)
        model=e.make(99203,'reference_bf16_math',cfg,DEVICE)
        with e.policy('reference_bf16_math'):
            logits,loss=e.forward_loss(model,x,y,'bf16',spec);self.assertEqual(logits.dtype,torch.bfloat16)
            loss.backward()
        self.assertTrue(all(p.dtype==torch.float32 and p.grad is not None and p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path)
    args=ap.parse_args();DEVICE=torch.device(args.device);e.q.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        e.save(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,skipped=len(result.skipped),device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
