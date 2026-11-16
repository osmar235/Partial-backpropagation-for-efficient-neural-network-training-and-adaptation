"""Autograd cuts, paused Adam, frozen BN, replay and native workload checks."""
import copy,json,sys,unittest
import torch
import stage12_adaptation as r
p=r.p
DEVICE='cuda' if '--cuda' in sys.argv else 'cpu'
if '--cuda' in sys.argv:sys.argv.remove('--cuda')

class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):p.execution.configure();cls.spec=json.loads(r.SPEC.read_text())

    def test_split_preserves_logits_and_active_gradients(self):
        model=p.make_model(9161,True).to(DEVICE);x=torch.randn(3,3,32,32,device=DEVICE);yy=torch.tensor([0,4,8],device=DEVICE)
        reference={};logits=None
        for mode in ['full','suffix','head']:
            model.zero_grad(set_to_none=True);z=p.forward(model,x,mode)
            if logits is None:logits=z.detach().clone()
            else:torch.testing.assert_close(z,logits,atol=0,rtol=0)
            torch.nn.functional.cross_entropy(z,yy).backward()
            for n,param in model.named_parameters():
                active=n.startswith('fc.') or mode=='full' or (mode=='suffix' and n.startswith('layer4.'))
                if mode=='full':reference[n]=param.grad.clone()
                elif active:torch.testing.assert_close(param.grad,reference[n],atol=1e-7,rtol=1e-5)
                else:self.assertIsNone(param.grad,n)

    def test_skipped_prefix_pauses_moments_weights_decay_and_bn(self):
        model=p.make_model(9161,True).to(DEVICE);opt=p.optimizer(model,self.spec)
        x=torch.randn(3,3,32,32,device=DEVICE);y=torch.tensor([0,4,8],device=DEVICE)
        p.update(model,opt,x,y,'full',0.)
        initial={n:t.clone() for n,t in model.state_dict().items()}
        prefix=p.groups(model)['prefix'];moments={id(t):copy.deepcopy(opt.state[t]) for t in prefix}
        for _ in range(3):p.update(model,opt,x,y,'suffix',0.)
        for n,t in model.named_parameters():
            if not n.startswith(('fc.','layer4.')):self.assertTrue(torch.equal(t,initial[n]),n)
        for t in prefix:
            for key,value in moments[id(t)].items():self.assertTrue(torch.equal(value,opt.state[t][key]))
        for n,t in model.named_buffers():self.assertTrue(torch.equal(t,initial[n]),n)
        p.update(model,opt,x,y,'full',0.)
        self.assertTrue(all(int(opt.state[t]['step'])==2 for t in prefix))
        self.assertTrue(all(int(opt.state[t]['step'])==5 for t in p.groups(model)['suffix']))

    def test_augmentation_replay_and_eval_normalization(self):
        x=torch.rand(3,3,32,32,device=DEVICE)
        a=p.transform(x,96,torch.Generator().manual_seed(1));b=p.transform(x,96,torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a,b));self.assertEqual(a.shape,(3,3,96,96))
        white=p.transform(torch.ones_like(x),96)
        torch.testing.assert_close(white.mean((0,2,3)),torch.tensor([(1-.485)/.229,(1-.456)/.224,(1-.406)/.225],device=DEVICE))

    def test_schedule_and_comparison_controls(self):
        spec=dict(self.spec,batches_per_epoch=2,epochs=4,head_warmup_epochs=1)
        self.assertEqual([p.mode_at('lp_periodic',i,spec) for i in range(8)],['head','head','full','suffix','suffix','suffix','full','suffix'])
        records={k:dict(final_accuracy=.92,total_run_seconds=100.,ready=True) for k in r.CONTROLS+['lp_periodic']}
        records['lp_periodic']['total_run_seconds']=80.
        self.assertTrue(r.screen(records,spec)['opportunity_pass'])
        records['lp_suffix']['total_run_seconds']=70.
        self.assertFalse(r.screen(records,spec)['speed_pass'])
        records['lp_suffix']['final_accuracy']=.91
        self.assertTrue(r.screen(records,spec)['opportunity_pass'])
        records['lp_periodic']['final_accuracy']=.9169
        self.assertFalse(r.screen(records,spec)['quality_pass'])
        records['full']['ready']=records['lp_full']['ready']=False
        self.assertFalse(r.screen(records,spec)['baseline_ready'])

    def test_native_pretrained_model_and_workload(self):
        model=p.make_model(9161,False).to(DEVICE)
        state=torch.load(p.WEIGHTS,map_location='cpu',weights_only=True)
        for n,t in model.state_dict().items():
            if n.startswith('fc.'):continue
            # Original published checkpoint predates this BatchNorm bookkeeping
            # buffer; torchvision's strict loader initializes it to zero.
            if n not in state:
                self.assertTrue(n.endswith('num_batches_tracked'));self.assertEqual(int(t),0)
            else:self.assertTrue(torch.equal(t.cpu(),state[n]),n)
        batch=128 if DEVICE=='cuda' else 2
        x=p.transform(torch.rand(batch,3,32,32,device=DEVICE),96)
        with torch.no_grad():torch.testing.assert_close(model(x),p.forward(model,x,'full'),atol=0,rtol=0)
        if DEVICE=='cuda':
            opt=p.optimizer(model,self.spec);buffers={n:t.clone() for n,t in model.named_buffers()}
            value=p.update(model,opt,x,torch.arange(batch,device=DEVICE)%10,'full',0.)
            self.assertTrue(value>0)
            self.assertTrue(all(torch.isfinite(t).all() for t in model.parameters()))
            for n,t in model.named_buffers():self.assertTrue(torch.equal(t,buffers[n]),n)

if __name__=='__main__':unittest.main(verbosity=2)
