"""Functional equivalence, optimizer pause and immutable-output checks."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
import torch
import pets_selective as p

DEVICE=torch.device('cpu')


class Tests(unittest.TestCase):
    def config(self):return p.configuration(True,True,DEVICE)

    def test_schedule(self):
        self.assertEqual(sum(p.active('periodic_r4',i) for i in range(690)),173)
        self.assertEqual(sum(p.active('full',i) for i in range(690)),690)
        self.assertEqual(sum(p.active('frozen',i) for i in range(690)),0)
        self.assertTrue(p.active('periodic_r4',48))
        self.assertFalse(p.active('periodic_r4',46))

    def test_full_matches_original_update(self):
        cfg=self.config();a=p.b.make_model(1,cfg,DEVICE);b=copy.deepcopy(a)
        x=torch.randn(4,3,16,16,device=DEVICE).contiguous(memory_format=torch.channels_last)
        y=torch.tensor([0,1,2,0],device=DEVICE)
        ao=p.b.optimizer(a,cfg['spec'],DEVICE);bo=p.b.optimizer(b,cfg['spec'],DEVICE)
        p.update(a,ao,x,y,'full',0,cfg,DEVICE)
        p.b.update(b,bo,x,y,'bf16',DEVICE,0)
        for name,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[name]),name)
        for ap,bp in zip(a.parameters(),b.parameters()):
            for key in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(ao.state[ap][key],bo.state[bp][key]))

    def test_skipped_step_preserves_momentum_and_decay(self):
        cfg=self.config();model=p.b.make_model(1,cfg,DEVICE);opt=p.b.optimizer(model,cfg['spec'],DEVICE)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        x=torch.randn(4,3,16,16,device=DEVICE);y=torch.tensor([0,1,2,0],device=DEVICE)
        p.update(model,opt,x,y,'periodic_r4',0,cfg,DEVICE)
        before={n:(v.detach().clone(),{k:s.clone() for k,s in opt.state[v].items()}) for n,v in model.named_parameters()}
        p.update(model,opt,x,y,'periodic_r4',1,cfg,DEVICE)
        for name,value in model.named_parameters():
            if p.is_prefix(name,True):
                self.assertIsNone(value.grad)
                self.assertTrue(torch.equal(value,before[name][0]))
                for key,s in opt.state[value].items():self.assertTrue(torch.equal(s,before[name][1][key]),key)
            else:self.assertEqual(int(opt.state[value]['step']),2)
        p.check_state(model,opt,initial,'periodic_r4',2,cfg)

    def test_frozen_never_initializes_state(self):
        cfg=self.config();model=p.b.make_model(1,cfg,DEVICE);opt=p.b.optimizer(model,cfg['spec'],DEVICE)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        x=torch.randn(4,3,16,16,device=DEVICE);y=torch.tensor([0,1,2,0],device=DEVICE)
        for i in range(5):p.update(model,opt,x,y,'frozen',i,cfg,DEVICE)
        p.check_state(model,opt,initial,'frozen',5,cfg)
        self.assertFalse(torch.equal(model.fc.weight.cpu(),initial['fc.weight']))

    def test_native_boundary_forward_and_suffix_gradients(self):
        cfg=p.configuration(True,False,DEVICE);a=p.b.make_model(1,cfg,DEVICE);b=copy.deepcopy(a)
        x=torch.randn(2,3,64,64,device=DEVICE).contiguous(memory_format=torch.channels_last)
        with p.b.arithmetic('bf16',DEVICE):
            full=p.forward(a,x,True);cut=p.forward(b,x,False)
        self.assertTrue(torch.equal(full,cut))
        full.float().square().sum().backward();cut.float().square().sum().backward()
        for (name,ap),(_,bp) in zip(a.named_parameters(),b.named_parameters()):
            if p.is_prefix(name):self.assertIsNone(bp.grad)
            else:self.assertTrue(torch.equal(ap.grad,bp.grad),name)

    def test_marker_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            parent=Path(td);attempt=parent/'attempt_x';attempt.mkdir();f=attempt/'summary.json';f.write_text('{}')
            cfg={'fingerprint':'x'}
            p.b.write(parent/'COMPLETE.json',dict(fingerprint='x',attempt='attempt_x',sha256={'summary.json':p.b.sha(f)}))
            self.assertEqual(p.b.verify_complete(parent,cfg),attempt)
            f.write_text('{"changed":true}')
            with self.assertRaises(AssertionError):p.b.verify_complete(parent,cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args();DEVICE=torch.device(args.device);p.b.configure(DEVICE)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:p.b.write(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
