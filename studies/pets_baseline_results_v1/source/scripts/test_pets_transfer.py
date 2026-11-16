"""Focused integrity checks for the independent-workload preparation."""
import tempfile
from pathlib import Path
import unittest
import torch
import pets_transfer as p

class Tests(unittest.TestCase):
    def test_split_and_stream(self):
        spec=p.read(p.SPEC);spec['classes']=3
        train,val=p.split_indices([0]*10+[1]*10+[2]*10,spec)
        self.assertEqual(len(train),24);self.assertEqual(len(val),6);self.assertFalse(set(train)&set(val))
        self.assertEqual(sorted(train+val),list(range(30)))
        a=p.stream(1,0,train,spec);b=p.stream(1,0,train,spec);c=p.stream(2,0,train,spec)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)));self.assertFalse(torch.equal(a[0],c[0]))

    def test_crop_exact(self):
        raw=torch.arange(3*6*6).reshape(1,3,6,6).byte();spec={'cache_size':6,'image_size':4}
        actual=p.transform(raw,spec,torch.device('cpu'),torch.tensor([[1,2]]),torch.tensor([True]))
        x=raw[:,:,1:5,2:6].flip(-1).float()/255
        expected=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        self.assertTrue(torch.equal(actual,expected))

    def test_pair_initial_and_state(self):
        device=torch.device('cpu');p.configure(device);cfg=p.configuration(True,True,device)
        a=p.make_model(1,cfg,device);b=p.make_model(1,cfg,device)
        self.assertEqual(p.state_hash(a.state_dict()),p.state_hash(b.state_dict()))
        before={k:v.clone() for k,v in a.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        opt=p.optimizer(a,cfg['spec'],device)
        p.update(a,opt,torch.randn(4,3,16,16),torch.tensor([0,1,2,0]),'fp32_tf32',device,0)
        self.assertTrue(all(int(s['step'])==1 for s in opt.state.values()))
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in before.items()))

    def test_marker_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            parent=Path(td);attempt=parent/'attempt_x';attempt.mkdir();f=attempt/'summary.json';f.write_text('{}')
            cfg={'fingerprint':'x'};p.write(parent/'COMPLETE.json',dict(fingerprint='x',attempt='attempt_x',sha256={'summary.json':p.sha(f)}))
            self.assertEqual(p.verify_complete(parent,cfg),attempt)
            f.write_text('{"changed":true}')
            with self.assertRaises(AssertionError):p.verify_complete(parent,cfg)

if __name__=='__main__':unittest.main()
