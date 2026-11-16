"""Artificial continuation, history mapping and local trajectory invariants."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
import stage13c_trajectory as r
from review_stage13C import screen
DEVICE=torch.device('cuda' if '--cuda' in sys.argv else 'cpu')
if '--cuda' in sys.argv: sys.argv.remove('--cuda')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.p.execution.configure()
        cls.recipe=dict(json.loads(r.b.prior.SPEC.read_text()),image_size=32,batch_size=3,eval_batch_size=4,batches_per_epoch=6)
        cls.spec=dict(json.loads(r.SPEC.read_text()),calibration_steps=4)
        cls.data=r.p.load_data(DEVICE,True)
        cls.cp,_,_,cls.trace=r.b.fake_source(9981,cls.data,cls.recipe,DEVICE)
        cls.cp=r.tensors_cpu(cls.cp)
        with tempfile.TemporaryDirectory() as folder:
            cls.cal,_,_=r.calibrate(9981,cls.cp,cls.data,cls.spec,cls.recipe,{},True,DEVICE,Path(folder)/'cal.pt')

    def test_full_replays_original_trajectory(self):
        model,opt,bg,ag=r.restore(9981,self.cp,self.cal,self.recipe,True,DEVICE,'full')
        for i,ix in enumerate(r.v.frozen6.epoch_batches(16,3,bg)):
            idx=ix.to(DEVICE); xx=r.p.transform(self.data[0][idx],32,ag)
            r.rates(opt,self.cp['updates']+i,self.recipe)
            values=r.l.step(model,opt,xx,self.data[1][idx],'full')
            self.assertEqual(values[-1],self.trace[i]['loss'])

    def test_moment_mapping_and_inactive_state(self):
        for arm in ('full','full_aux','local','suffix'):
            model,opt,_,_=r.restore(9981,self.cp,self.cal,self.recipe,True,DEVICE,arm)
            ages=r.b.ages(model,opt)
            for n,age in ages.items():
                expected=12 if n.startswith('net.fc.') else 4 if n.startswith('aux') and arm in ('local','full_aux') else 0
                self.assertEqual(age,expected,n)
            initial={n:t.clone() for n,t in model.state_dict().items()}
            xx=r.p.transform(self.data[0][:3],32)
            for _ in range(3): r.l.step(model,opt,xx,self.data[1][:3],arm)
            for n,t in model.named_parameters():
                self.assertEqual(r.b.ages(model,opt)[n],ages[n]+(3 if r.l.active(n,arm) else 0))
                if not r.l.active(n,arm):
                    self.assertTrue(torch.equal(t,initial[n])); self.assertIsNone(t.grad); self.assertNotIn(t,opt.state)
            for n,t in model.named_buffers(): self.assertTrue(torch.equal(t,initial[n]))

    def test_local_final_block_agrees_at_start(self):
        gradients={}
        for arm in ('full','full_aux','local','suffix'):
            model,opt,_,_=r.restore(9981,self.cp,self.cal,self.recipe,True,DEVICE,arm)
            r.l.backward(model,r.p.transform(self.data[0][:3],32),self.data[1][:3],arm)
            gradients[arm]={n:t.grad.clone() for n,t in model.named_parameters() if r.l.group(n)=='block2'}
        for arm in gradients:
            for n,t in gradients[arm].items(): torch.testing.assert_close(t,gradients['full'][n],atol=0,rtol=0)

    def test_calibration_does_not_change_source(self):
        self.assertEqual(r.old.state_hash({n[4:]:t for n,t in self.cal['model_state'].items() if n.startswith('net.')}),r.old.state_hash(self.cp['model_state']))
        with tempfile.TemporaryDirectory() as folder:
            other,_,_=r.calibrate(9981,self.cp,self.data,self.spec,self.recipe,{},True,DEVICE,Path(folder)/'again.pt')
        self.assertTrue(r.equal_state(self.cal,other))

    def test_cost_gate_charges_calibration_and_controls(self):
        records={arm:dict(accuracy=.89,loss=.3,initial_accuracy=.81,initial_loss=.55,
            charged_training_seconds=10.,max_incremental_allocated=100.) for arm in ('full','full_aux','local','suffix')}
        records['local']['charged_training_seconds']=8.
        self.assertTrue(screen(records,self.spec,True,False)['seed_screen_pass'])
        records['local']['charged_training_seconds']=9.1
        self.assertFalse(screen(records,self.spec,True,False)['seed_screen_pass'])
        records['local']['max_incremental_allocated']=80.
        self.assertTrue(screen(records,self.spec,True,False)['seed_screen_pass'])
        records['suffix']['max_incremental_allocated']=90.
        self.assertFalse(screen(records,self.spec,True,False)['seed_screen_pass'])
        records['local']['charged_training_seconds']=7.
        self.assertFalse(screen(records,self.spec,False,False)['seed_screen_pass'])
        self.assertFalse(screen(records,self.spec,True,True)['seed_screen_pass'])
        records['local']['accuracy']=.8869
        self.assertFalse(screen(records,self.spec,True,False)['seed_screen_pass'])

    def test_native_restored_full_and_local_final_block(self):
        net=r.p.make_model(9981,False).to(DEVICE); original_opt=r.p.optimizer(net,self.recipe)
        xx=r.p.transform(self.data[0][:2],96); yy=self.data[1][:2]
        r.p.update(net,original_opt,xx,yy,'head',0.)
        cp=r.tensors_cpu(dict(model_state=net.state_dict(),optimizer_state=original_opt.state_dict(),
            batch_rng=torch.Generator().get_state(),augmentation_rng=torch.Generator().get_state()))
        local=r.l.LocalModel(net,9981,False).to(DEVICE)
        auxiliary=torch.optim.AdamW(list(local.aux0.parameters())+list(local.aux1.parameters()),lr=.001,weight_decay=.01)
        with torch.no_grad(): z0=local.block0(xx); z1=local.block1(z0)
        objective=torch.nn.functional.cross_entropy(local.auxiliary(z0,0),yy)+torch.nn.functional.cross_entropy(local.auxiliary(z1,1),yy)
        objective.backward(); auxiliary.step(); local.zero_grad(set_to_none=True)
        calibrated=r.tensors_cpu(dict(model_state=local.state_dict(),aux_optimizer_state=auxiliary.state_dict()))
        full,fo,_,_=r.restore(9981,cp,calibrated,self.recipe,False,DEVICE,'full')
        decoupled,lo,_,_=r.restore(9981,cp,calibrated,self.recipe,False,DEVICE,'local')
        for n,t in decoupled.named_parameters():
            if n.startswith('aux'):
                other=dict(local.named_parameters())[n]
                self.assertTrue(r.equal_state(lo.state[t],auxiliary.state[other]))
        r.rates(fo,704,self.recipe); r.rates(lo,704,self.recipe); r.b.rates(original_opt,704,self.recipe)
        r.p.update(net,original_opt,xx,yy,'full',0.)
        r.l.step(full,fo,xx,yy,'full'); r.l.step(decoupled,lo,xx,yy,'local')
        self.assertEqual(r.old.state_hash(net.state_dict()),r.old.state_hash(full.net.state_dict()))
        for n,t in full.named_parameters():
            if r.l.group(n)=='block2': torch.testing.assert_close(t,dict(decoupled.named_parameters())[n],atol=0,rtol=0)


if __name__=='__main__': unittest.main(verbosity=2)
