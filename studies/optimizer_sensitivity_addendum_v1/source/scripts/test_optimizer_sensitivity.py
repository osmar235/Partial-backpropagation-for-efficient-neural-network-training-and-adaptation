"""Tests for matched states, sign/zero bins, actual AdamW and precise advances."""
import argparse
import copy
from pathlib import Path
import unittest
import torch
import optimizer_sensitivity as d

DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def setUp(self):
        self.spec=d.read(d.SPEC)
        self.weights={'weight':torch.tensor([1.,1.,1.,1.])}
        self.state=d.cpu(d.optimizer([torch.nn.Parameter(self.weights['weight'].clone())],self.spec).state_dict())

    def test_signed_small_gradient_and_moment_history(self):
        a={'weight':torch.tensor([1e-6,0.,1e-4,1e-2])};b={'weight':torch.tensor([-1e-6,1e-8,1e-4,1e-2])}
        ra,_,_=d.replay(self.weights,self.state,a,self.spec,DEVICE)
        rb,_,_=d.replay(self.weights,self.state,b,self.spec,DEVICE)
        result=d.analyze(self.weights,self.state,a,b,ra['delta'],rb['delta'],self.spec)
        self.assertEqual(result['strict_sign_flips'],1);self.assertEqual(result['zero_disagreements'],1)
        self.assertEqual(sum(z['coordinates'] for z in result['bins']),4)
        self.assertGreater(result['update']['max_abs'],1.9e-4)
        for coord in result['top_coordinates']:
            self.assertAlmostEqual(coord['reference_delta'],coord['reference_analytic_delta'],delta=1.3e-7)
            self.assertAlmostEqual(coord['candidate_delta'],coord['candidate_analytic_delta'],delta=1.3e-7)
        retained=copy.deepcopy(self.state);retained['state'][0]={'step':torch.tensor(100.),'exp_avg':torch.full((4,),.1),'exp_avg_sq':torch.full((4,),.1)}
        sa,_,_=d.replay(self.weights,retained,a,self.spec,DEVICE);sb,_,_=d.replay(self.weights,retained,b,self.spec,DEVICE)
        stable=d.analyze(self.weights,retained,a,b,sa['delta'],sb['delta'],self.spec)
        self.assertLess(stable['update']['max_abs'],result['update']['max_abs'])
        self.assertEqual(d.opt_hash(self.state),ra['start_optimizer_sha256']);self.assertFalse(self.state['state'])

    def test_zero_self_comparison_and_corrupt_state(self):
        a={'weight':torch.zeros(4)};record,_,_=d.replay(self.weights,self.state,a,self.spec,DEVICE)
        result=d.analyze(self.weights,self.state,a,a,record['delta'],record['delta'],self.spec)
        self.assertEqual(result['gradient']['max_abs'],0);self.assertEqual(result['update']['max_abs'],0)
        self.assertEqual(result['gradient']['cosine'],1)
        malformed=copy.deepcopy(self.state);malformed['param_groups'][0]['params']=[1]
        with self.assertRaises(AssertionError):d.replay(self.weights,malformed,a,self.spec,DEVICE)

    def test_same_gradient_repeat_and_actual_parameter_return(self):
        a={'weight':torch.tensor([-1e-6,1e-6,0.,.01])}
        ra,oa,wa=d.replay(self.weights,self.state,a,self.spec,DEVICE);rb,ob,wb=d.replay(self.weights,self.state,a,self.spec,DEVICE)
        self.assertEqual(d.opt_hash(oa),d.opt_hash(ob));self.assertTrue(torch.equal(wa['weight'],wb['weight']))
        self.assertTrue(torch.equal(wa['weight']-self.weights['weight'],ra['delta']['weight']))
        self.assertEqual(ra['optimizer_age'],1)

    def test_capture_and_trained_fixture(self):
        cfg=d.configuration(True,DEVICE);task=next(t for t in d.trajectories(cfg) if t['kind']=='trained')
        x,y,_=d.inputs(task,cfg,DEVICE);weights,fresh,history,source=d.load_origin(task,cfg,x,y,DEVICE)
        self.assertEqual(source['kind'],'artificial_trained_fixture');self.assertFalse(fresh['state']);d.validate_state(history,weights,2)
        arm='reference_bf16_math' if DEVICE.type=='cuda' else 'reference_fp32_math'
        cap=d.capture(weights,task,arm,cfg,x,y,DEVICE);self.assertEqual(cap['status'],'FINITE')
        self.assertTrue(all(g.dtype==torch.float32 for g in cap['gradient'].values()))
        self.assertTrue(cap['policy']['deterministic']);self.assertEqual(cap['dispatch']['math'],2)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path)
    args=ap.parse_args();DEVICE=torch.device(args.device);d.e.q.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();d.save(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
