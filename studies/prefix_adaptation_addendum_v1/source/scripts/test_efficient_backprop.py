"""Small artificial tests of identity, backend isolation and rejection logic."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import torch
import efficient_backprop as e
DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def test_backend_context_restores_after_exception(self):
        before=e.m.execution_settings()
        with self.assertRaisesRegex(ValueError,'test cleanup'):
            with e.sdpa_kernel(e.BACKENDS['efficient']):
                self.assertFalse(torch.backends.cuda.math_sdp_enabled())
                self.assertTrue(torch.backends.cuda.mem_efficient_sdp_enabled())
                raise ValueError('test cleanup')
        self.assertEqual(e.m.execution_settings(),before)

    def test_math_identity_and_numerical_rejection(self):
        spec=json.loads(e.SPEC.read_text());gen=torch.Generator().manual_seed(99141)
        x=torch.randn(3,3,32,32,generator=gen).to(DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        a=e.snapshots(99141,'math',spec,True,x,y,DEVICE);b=e.snapshots(99141,'math',spec,True,x,y,DEVICE)
        good=e.compare(a,b,spec['tolerances']);self.assertEqual(good['status'],'PASS')
        self.assertTrue(all(z['max_abs']==0 for z in good['metrics']))
        name=next(iter(b[0]['gradient']));b[0]['gradient'][name].flatten()[0]+=1
        self.assertEqual(e.compare(a,b,spec['tolerances'])['status'],'FAIL_NUMERICAL')
        c=copy.deepcopy(a);c[0]['rng']['cpu']='changed'
        self.assertEqual(e.compare(a,c,spec['tolerances'])['status'],'FAIL_NUMERICAL')

    def test_math_dispatch_and_full_gradients(self):
        spec=json.loads(e.SPEC.read_text());x=torch.ones(3,3,32,32,device=DEVICE);y=torch.tensor([1,2,3],device=DEVICE)
        result=e.probe(99142,3,'math',spec,True,x,y,DEVICE)
        self.assertEqual(result['status'],'SUPPORTED');self.assertEqual(result['attention_operators'][e.OPS['math']],2)
        self.assertTrue(result['all_parameters_have_finite_full_gradients'])

    def test_unknown_errors_are_not_unsupported(self):
        self.assertTrue(e.unsupported(RuntimeError('No available kernel. Aborting execution.')))
        self.assertFalse(e.unsupported(RuntimeError('CUDA out of memory')))
        self.assertFalse(e.unsupported(RuntimeError('mat1 and mat2 shapes cannot be multiplied')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device('cuda' if args.cuda else 'cpu');e.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',device=str(DEVICE),tests=result.testsRun),indent=2)+'\n',encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
