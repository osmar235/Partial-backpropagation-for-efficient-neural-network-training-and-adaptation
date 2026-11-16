"""Transform identities, fair controls, full learning and deterministic replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy, sys, unittest
import torch
import torch.nn.functional as F
import stage10_models as m
from stage10_representation import update
from review_stage10A import target

DEVICE='cuda' if '--cuda' in sys.argv else 'cpu'
if '--cuda' in sys.argv:sys.argv.remove('--cuda')

class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m.configure()
        if DEVICE=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA rehearsal unavailable')

    def test_haar_inverse_energy_linearity_and_coarse_identity(self):
        torch.manual_seed(301);x=torch.randn(3,3,32,32,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x)
        ll,details=m.haar(x);actual=m.inverse_haar(ll,details)
        torch.testing.assert_close(actual,x,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(ll.square().sum()+details.square().sum(),x.square().sum(),rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(.5*ll,F.avg_pool2d(x,2)))
        aa,bb=m.haar(.3*x+.7*y);yl,yd=m.haar(y)
        torch.testing.assert_close(aa,.3*ll+.7*yl,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(bb,.3*details+.7*yd,rtol=1e-12,atol=1e-12)
        actual.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x),rtol=0,atol=0)

    def test_pixel_grouping_is_lossless(self):
        x=torch.arange(3*32*32).view(1,3,32,32).float();a,others=m.pixel_groups(x);b,c,d=others.chunk(3,1)
        restored=torch.empty_like(x);restored[:,:,::2,::2]=a;restored[:,:,::2,1::2]=b
        restored[:,:,1::2,::2]=c;restored[:,:,1::2,1::2]=d
        self.assertTrue(torch.equal(x,restored))

    def test_controls_share_components_and_capacity(self):
        models={v:m.make_model(v,1401,False) for v in m.VARIANTS}
        ref=models['coarse_rgb16'].body.state_dict()
        for variant in ['pixel_branches','haar_branches']:
            for name,tensor in ref.items():self.assertTrue(torch.equal(tensor,models[variant].body.state_dict()[name]))
        for name,tensor in models['pixel_branches'].state_dict().items():
            self.assertTrue(torch.equal(tensor,models['haar_branches'].state_dict()[name]))
        counts={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
        self.assertEqual(counts['pixel_branches'],counts['haar_branches'])
        self.assertLess(counts['small_rgb32'],counts['full_rgb32']);self.assertGreater(counts['haar_branches'],counts['coarse_rgb16'])

    def test_details_can_change_prediction_when_coarse_is_identical(self):
        model=m.make_model('haar_branches',1401,True).eval();torch.nn.init.normal_(model.head.weight,std=.02)
        a=torch.zeros(2,3,32,32);b=a.clone();b[:,:,::2,::2]=1;b[:,:,1::2,1::2]=1
        b[:,:,::2,1::2]=-1;b[:,:,1::2,::2]=-1
        self.assertTrue(torch.equal(F.avg_pool2d(a,2),F.avg_pool2d(b,2)))
        self.assertGreater(float((model(a)-model(b)).detach().abs().max()),1e-6)

    def test_full_updates_and_bitwise_replay(self):
        torch.manual_seed(303);x=torch.randn(3,3,32,32,device=DEVICE);labels=torch.tensor([0,1,2],device=DEVICE)
        targets=F.one_hot(labels,10).float()
        cases=[(name,True) for name in m.VARIANTS]+[('haar_branches',False)]
        for variant,fake in cases:
            snapshots=[]
            for _ in range(2):
                model=m.make_model(variant,1401,fake).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
                losses=[update(model,opt,x,targets,30001+i) for i in range(4)]
                for param in model.parameters():self.assertIsNotNone(param.grad);self.assertEqual(int(opt.state[param]['step']),4)
                if model.dual:self.assertTrue(any(float(p.grad.norm())>0 for p in model.detail.parameters()))
                snapshots.append((losses,{n:t.detach().cpu().clone() for n,t in model.state_dict().items()},copy.deepcopy(opt.state_dict())))
            self.assertEqual(snapshots[0][0],snapshots[1][0],variant)
            for name,tensor in snapshots[0][1].items():self.assertTrue(torch.equal(tensor,snapshots[1][1][name]),(variant,name))
            for i,state in snapshots[0][2]['state'].items():
                for name,tensor in state.items():self.assertTrue(torch.equal(tensor,snapshots[1][2]['state'][i][name]),(variant,i,name))

    def test_sustained_target_and_censoring(self):
        rows=[dict(epoch=i+1,validation_accuracy=x) for i,x in enumerate([.79,.80,.78,.81,.82,.83])]
        first,last=target(rows,.8,3);self.assertEqual(first['epoch'],2);self.assertEqual(last['epoch'],6)
        first,last=target(rows,.9,3);self.assertIsNone(first);self.assertIsNone(last)

if __name__=='__main__':unittest.main(verbosity=2)
