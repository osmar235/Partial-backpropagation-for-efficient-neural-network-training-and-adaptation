"""Artificial invariance checks for profiling annotations and phase timing."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import torch
import runtime_profile as r

DEVICE=torch.device('cpu')

class ProfileTests(unittest.TestCase):
    def test_phase_wrapper_matches_original(self):
        spec=json.loads(r.SPEC.read_text());gen=torch.Generator().manual_seed(891)
        x=torch.randn(3,3,32,32,generator=gen).to(DEVICE);y=torch.tensor([0,1,2],device=DEVICE)
        for arm in spec['arms']:
            native=r.cost.make(99111,arm,spec,True).to(DEVICE);wrapped=copy.deepcopy(native)
            opt=torch.optim.AdamW(native.parameters(),lr=.0001,weight_decay=.05)
            other=torch.optim.AdamW(wrapped.parameters(),lr=.0001,weight_decay=.05)
            native.train();wrapped.train();rec=r.PhaseRecorder('events',DEVICE)
            for step in range(2):
                expected=r.cost.update(native,opt,x,y,99111,step,spec)
                cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state() if DEVICE.type=='cuda' else None
                actual=r.instrumented_update(wrapped,other,x,y,99111,step,spec,rec)
                self.assertEqual(actual,expected);self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
                if cuda is not None:self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))
                self.assertEqual(r.old.state_hash(native.state_dict()),r.old.state_hash(wrapped.state_dict()))
                self.assertEqual(r.optimizer_hash(opt),r.optimizer_hash(other))
            rows=rec.rows();self.assertEqual(len(rows),10)
            self.assertTrue(all(row['host_seconds']>=0 for row in rows))
            self.assertTrue(all(row['cuda_stream_ms'] is not None for row in rows) if DEVICE.type=='cuda' else all(row['cuda_stream_ms'] is None for row in rows))

    def test_module_hooks_preserve_forward_and_cleanup(self):
        spec=json.loads(r.SPEC.read_text());model=r.cost.make(99112,'sampled_quarter',spec,True).to(DEVICE);model.train()
        x=torch.ones(2,3,32,32,device=DEVICE)
        torch.manual_seed(771);expected=model(x);state=torch.get_rng_state()
        torch.manual_seed(771)
        with r.module_ranges(model):actual=model(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertTrue(torch.equal(torch.get_rng_state(),state))
        with self.assertRaisesRegex(RuntimeError,'cleanup test'):
            with r.module_ranges(model):raise RuntimeError('cleanup test')
        for module in model.modules():self.assertFalse(module._forward_hooks or module._forward_pre_hooks)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');r.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileTests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',device=str(DEVICE),
            tests=result.testsRun,artificial_only=True),indent=2)+'\n',encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
