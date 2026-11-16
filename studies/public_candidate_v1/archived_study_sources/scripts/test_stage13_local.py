"""Local-gradient isolation, synchronous equivalence and optimizer semantics."""
import copy,json,sys,unittest
import torch
import torch.nn.functional as F
import stage13_diagnostic as r
l=r.l;p=r.p
DEVICE='cuda' if '--cuda' in sys.argv else 'cpu'
if '--cuda' in sys.argv:sys.argv.remove('--cuda')
class LocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):p.execution.configure();cls.spec=json.loads(r.SPEC.read_text())
    def model(self,fake=True):return l.LocalModel(p.make_model(9951,fake),9951,fake).to(DEVICE)

    def test_local_gradients_equal_independent_block_objectives(self):
        model=self.model();x=torch.randn(3,3,32,32,device=DEVICE);y=torch.tensor([0,4,8],device=DEVICE)
        l.backward(model,x,y,'local');actual={n:t.grad.clone() for n,t in model.named_parameters()}
        for block in ['block0','block1','block2']:
            model.zero_grad(set_to_none=True)
            if block=='block0':loss=F.cross_entropy(model.auxiliary(model.block0(x),0),y)
            elif block=='block1':
                with torch.no_grad():z=model.block0(x)
                loss=F.cross_entropy(model.auxiliary(model.block1(z),1),y)
            else:
                with torch.no_grad():z=model.block1(model.block0(x))
                loss=F.cross_entropy(model.block2(z),y)
            loss.backward()
            for n,t in model.named_parameters():
                affected=l.group(n)==block or (block=='block0' and n.startswith('aux0')) or (block=='block1' and n.startswith('aux1'))
                if affected:torch.testing.assert_close(t.grad,actual[n],atol=1e-7,rtol=1e-5)
                else:self.assertIsNone(t.grad,n)

    def test_final_gradient_preserved_and_auxiliary_control_changes_early_gradient(self):
        model=self.model();x=torch.randn(3,3,32,32,device=DEVICE);y=torch.tensor([0,4,8],device=DEVICE);grads={};losses={}
        for mode in l.MODES:
            losses[mode]=l.backward(model,x,y,mode);grads[mode]=l.gradient_vectors(model)
        for mode in l.MODES:
            self.assertAlmostEqual(losses[mode][-1],losses['full'][-1],places=7)
            torch.testing.assert_close(grads[mode]['block2'],grads['full']['block2'],atol=1e-7,rtol=1e-5)
        self.assertGreater(float((grads['local']['block0']-grads['full_aux']['block0']).abs().max()),1e-8)
        self.assertEqual(float(grads['suffix']['block0'].abs().sum()),0.)

    def test_zero_moment_reset_matches_fresh_and_inactive_parameters_pause(self):
        model=self.model();initial=copy.deepcopy(model.state_dict());x=torch.randn(3,3,32,32,device=DEVICE);y=torch.tensor([0,4,8],device=DEVICE)
        for mode in l.MODES:
            model.load_state_dict(initial);opt=l.optimizer(model,self.spec);l.step(model,opt,x,y,mode)
            l.verify_step(model,opt,{n:t.cpu() for n,t in initial.items()},mode)
            expected=copy.deepcopy(model.state_dict());model.load_state_dict(initial);l.zero_optimizer(opt)
            l.step(model,opt,x,y,mode)
            for n,t in model.state_dict().items():self.assertTrue(torch.equal(t,expected[n]),(mode,n))

    def test_disjoint_pool_and_stream_replay(self):
        spec=dict(self.spec,calibration_per_class=2,probe_per_class=2,score_per_class=2,calibration_steps=4,probe_trials=4,batch_size=3)
        y=torch.arange(10).repeat_interleave(6);a=r.streams(y,9951,spec);self.assertEqual(a,r.streams(y,9951,spec))
        pools=[set(a['pools'][key]) for key in ['calibration','probe','score']]
        for i,j in [(0,1),(0,2),(1,2)]:self.assertFalse(pools[i]&pools[j])

    def test_native_resnet_local_update_and_final_gradient(self):
        model=self.model(False);x=p.transform(torch.rand(4,3,32,32,device=DEVICE),96);y=torch.tensor([0,4,8,1],device=DEVICE)
        l.backward(model,x,y,'full');reference=l.gradient_vectors(model)['block2']
        initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};opt=l.optimizer(model,self.spec)
        l.step(model,opt,x,y,'local');l.verify_step(model,opt,initial,'local')
        torch.testing.assert_close(l.gradient_vectors(model)['block2'],reference,atol=1e-6,rtol=1e-4)
        self.assertTrue(any(not torch.equal(t.detach().cpu(),initial[n]) for n,t in model.named_parameters() if l.group(n)=='block0'))

if __name__=='__main__':unittest.main(verbosity=2)
