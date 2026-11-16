"""Meaningful integration/mechanics tests; artificial inputs only."""
import copy, itertools, json, tempfile, unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
import stage8_vit_v2 as v
import stage8_workflow_v2 as w

torch.set_num_threads(1)

class Mechanics(unittest.TestCase):
    def setUp(self):v.configure();v.seed_all(9201)

    def test_native_model_and_partition(self):
        model=v.make_model();prefix,suffix,names=v.partition(model,2)
        self.assertEqual(model(torch.rand(2,3,32,32)).shape,(2,10))
        self.assertIn('encoder.pos_embedding',names)
        self.assertIn('encoder.layers.encoder_layer_1.ln_1.weight',names)
        self.assertNotIn('encoder.layers.encoder_layer_2.ln_1.weight',names)
        self.assertEqual(sum(p.numel() for p in prefix+suffix),sum(p.numel() for p in model.parameters()))
        self.assertFalse(any(isinstance(m,nn.BatchNorm2d) for m in model.modules()))
        model.eval();nn.init.normal_(model.heads.head.weight,std=.02)
        loss=F.cross_entropy(model(torch.rand(2,3,32,32)),torch.tensor([0,1]))
        full=torch.autograd.grad(loss,list(model.parameters()),retain_graph=True)
        selected=torch.autograd.grad(loss,suffix)
        mapping={id(p):g for p,g in zip(model.parameters(),full)}
        for p,g in zip(suffix,selected):self.assertTrue(torch.equal(g,mapping[id(p)]))

    def test_exact_suffix_and_continuation(self):
        model=v.make_model(tiny=True);model.eval()
        nn.init.normal_(model.heads.head.weight,std=.02)
        prefix,suffix,_=v.partition(model,1)
        x=torch.rand(2,3,32,32);y=torch.tensor([0,1])
        x,y,_=v.mix_batch(x,y,v.np.random.default_rng(12001),torch.Generator().manual_seed(12002),.2,10)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
        opt.zero_grad(set_to_none=True)
        v.update(model,opt,F.cross_entropy(model(x),y),prefix,suffix,False,'lgpb',None)
        ref=copy.deepcopy(model);refopt=torch.optim.AdamW(ref.parameters(),lr=.001,weight_decay=.05)
        refopt.load_state_dict(copy.deepcopy(opt.state_dict()))
        refprefix,refsuffix,_=v.partition(ref,1)
        opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x),y)
        full=torch.autograd.grad(loss,list(model.parameters()),retain_graph=True)
        mapping={id(p):g for p,g in zip(model.parameters(),full)}
        v.compute_backward(loss,prefix,suffix,True,'lgpb')
        for p in suffix:self.assertTrue(torch.equal(p.grad,mapping[id(p)]))
        for p in prefix:self.assertTrue(torch.equal(p.grad,torch.zeros_like(p)))
        F.cross_entropy(ref(x),y).backward()
        for p in refprefix:p.grad.zero_()
        prior=[p.detach().clone() for p in prefix]
        v.optimizer_step(opt,prefix,True,'lgpb',None);refopt.step()
        for p,r in zip(model.parameters(),ref.parameters()):self.assertTrue(torch.equal(p,r))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(prefix,prior)))
        self.assertTrue(all(int(opt.state[p]['step'])==2 for p in prefix))

    def test_replay_keeps_prefix_state_and_exact_delta(self):
        model=v.make_model(tiny=True);model.eval();nn.init.normal_(model.heads.head.weight,std=.02)
        prefix,suffix,_=v.partition(model,1)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
        x=torch.rand(2,3,32,32);y=torch.tensor([0,1])
        before=[p.detach().clone() for p in prefix]
        delta=v.update(model,opt,F.cross_entropy(model(x),y),prefix,suffix,False,'replay_prefix',None)
        after=[p.detach().clone() for p in prefix]
        for b,a,d in zip(before,after,delta):self.assertTrue(torch.equal(a-b,d))
        state=copy.deepcopy([opt.state[p] for p in prefix]);opt.zero_grad(set_to_none=True)
        v.update(model,opt,F.cross_entropy(model(x),y),prefix,suffix,True,'replay_prefix',delta)
        for p,a,d,s in zip(prefix,after,delta,state):
            self.assertTrue(torch.equal(p,a+d))
            for key,value in s.items():self.assertTrue(torch.equal(opt.state[p][key],value))

    def test_validation_never_loads_test_images(self):
        class ArtificialDataset:
            def __init__(self,root,train,download):
                if not train:raise AssertionError('Development touched the CIFAR test split')
                self.data=v.np.zeros((50000,32,32,3),dtype=v.np.uint8)
                self.targets=[c for c in range(10) for _ in range(5000)]
        with patch.object(v,'CIFAR10',ArtificialDataset):
            x,y,ev,ey,meta=v.load_data('development',torch.device('cpu'))
        self.assertEqual((len(y),len(ey)),(45000,5000))
        self.assertEqual(torch.bincount(ey).tolist(),[500]*10)
        tr,va=v.split_indices([c for c in range(10) for _ in range(5000)])
        self.assertFalse(set(tr.tolist())&set(va.tolist()))

    def test_rate_schedule_orders_and_masks(self):
        spec=v.SPEC;total=196*spec['epochs']
        self.assertAlmostEqual(v.learning_rate(0,total,196,.001,spec),.001/(196*5))
        self.assertAlmostEqual(v.learning_rate(196*5-1,total,196,.001,spec),.001)
        self.assertAlmostEqual(v.learning_rate(total-1,total,196,.001,spec),.00001)
        for a,b in itertools.combinations(spec['variants'],2):
            self.assertEqual(sum(row.index(a)<row.index(b) for row in spec['evaluation_orders']),3)
        for variant in spec['variants']:
            self.assertEqual(sum(order.index(variant)<2 for order in spec['evaluation_orders']),3)
        profile=[.0,.5,.3599]*50;mask=v.periodic_indices(150,196,profile)
        self.assertEqual(len(mask),sum(round(196*q) for q in profile))
        self.assertTrue(all(j+1 not in mask for j in mask))

    def test_resume_and_fingerprint_rejection(self):
        spec=copy.deepcopy(v.SPEC);spec.update(epochs=1,batch_size=4,eval_batch_size=4,
            algorithm_warmup_steps=1,lr_warmup_epochs=0)
        device=torch.device('cpu');data=v.load_data('development',device,fake=True)
        with tempfile.TemporaryDirectory(prefix='stage8_unit_') as tmp:
            out=Path(tmp)
            first=v.run_trajectory(out,'development',9003,'lgpb',.001,device,data,spec,tiny=True,fake=True)
            again=v.run_trajectory(out,'development',9003,'lgpb',.001,device,data,spec,tiny=True,fake=True)
            self.assertEqual(first,again)
            saved=torch.load(out/'final_checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(saved['fingerprint'],first['fingerprint'])
            self.assertEqual(first['scalar_loss_item_calls'],4)
            with self.assertRaises(ValueError):
                v.run_trajectory(out,'development',9003,'lgpb',.0003,device,data,spec,tiny=True,fake=True)

    def test_evaluation_requires_prior_freeze(self):
        with tempfile.TemporaryDirectory(prefix='stage8_gate_') as tmp:
            args=SimpleNamespace(freeze_file=Path(tmp)/'not_created.json')
            with patch.object(v,'load_data') as loader:
                with self.assertRaises(FileNotFoundError):w.evaluation(args,v.SPEC,torch.device('cpu'))
                loader.assert_not_called()

    def test_poor_baseline_blocks_freezing(self):
        with tempfile.TemporaryDirectory(prefix='stage8_quality_') as tmp:
            with patch.object(w,'completed',return_value={'final_eval_accuracy':.7}):
                with self.assertRaises(RuntimeError):w.select_baseline(Path(tmp),v.SPEC,False)
            self.assertFalse(json.loads((Path(tmp)/'baseline_selection.json').read_text())['ready'])

    def test_both_readiness_thresholds_remain_enforced(self):
        # First pair passes mean but fails minimum; second passes minimum but fails mean.
        for accuracies in ((.8299,.8701),(.849,.85)):
            with tempfile.TemporaryDirectory(prefix='stage8_quality_') as tmp:
                with patch.object(w,'completed',side_effect=[{'final_eval_accuracy':a} for a in accuracies]):
                    with self.assertRaises(w.BaselineReadinessError):w.select_baseline(Path(tmp),v.SPEC,False)
        with tempfile.TemporaryDirectory(prefix='stage8_quality_') as tmp:
            with patch.object(w,'completed',side_effect=[{'final_eval_accuracy':a} for a in (.84,.86)]):
                self.assertTrue(w.select_baseline(Path(tmp),v.SPEC,False)['ready'])

    def test_mixup_convexity_soft_loss_and_rng_isolation(self):
        x=torch.arange(24,dtype=torch.float32).reshape(4,1,2,3);y=torch.tensor([0,1,2,3])
        global_torch=torch.get_rng_state().clone()
        global_numpy=copy.deepcopy(v.np.random.get_state())
        rng=v.np.random.default_rng(123);perm=torch.Generator().manual_seed(456)
        expected_lambda=float(v.np.random.default_rng(123).beta(.2,.2))
        expected_perm=torch.randperm(4,generator=torch.Generator().manual_seed(456))
        mixed,targets,lam=v.mix_batch(x,y,rng,perm,.2,10)
        self.assertEqual(lam,expected_lambda)
        self.assertTrue(torch.equal(mixed,lam*x+(1-lam)*x[expected_perm]))
        hard=F.one_hot(y,10).float()
        self.assertTrue(torch.equal(targets,lam*hard+(1-lam)*hard[expected_perm]))
        self.assertTrue(torch.allclose(targets.sum(1),torch.ones(4)))
        self.assertTrue(bool((targets>=0).all()))
        self.assertTrue(torch.equal(global_torch,torch.get_rng_state()))
        after=v.np.random.get_state()
        self.assertEqual(global_numpy[0],after[0]);self.assertTrue(v.np.array_equal(global_numpy[1],after[1]))
        self.assertEqual(global_numpy[2:],after[2:])
        logits=torch.randn(4,10,requires_grad=True)
        soft=F.cross_entropy(logits,targets,label_smoothing=.1)
        weighted=lam*F.cross_entropy(logits,y,label_smoothing=.1)+(1-lam)*F.cross_entropy(logits,y[expected_perm],label_smoothing=.1)
        self.assertTrue(torch.allclose(soft,weighted,atol=1e-6,rtol=1e-6))
        soft_grad=torch.autograd.grad(soft,logits,retain_graph=True)[0]
        self.assertTrue(torch.allclose(soft_grad,torch.autograd.grad(weighted,logits)[0],atol=1e-7,rtol=1e-6))
        # Identical independent streams yield the same entire batch sequence for paired arms.
        streams=[(v.np.random.default_rng(789),torch.Generator().manual_seed(987)) for _ in range(4)]
        for _ in range(8):
            batches=[v.mix_batch(x,y,r,p,.2,10) for r,p in streams]
            for candidate in batches[1:]:
                self.assertTrue(torch.equal(batches[0][0],candidate[0]))
                self.assertTrue(torch.equal(batches[0][1],candidate[1]))
                self.assertEqual(batches[0][2],candidate[2])

    def test_busy_gpu_guard(self):
        with patch.object(v,'telemetry',return_value={'available':True,'gpus':[{'utilization.gpu':'90'}]}):
            with patch.object(v.time,'sleep'):
                with self.assertRaisesRegex(RuntimeError,'already busy'):v.idle_preflight()

if __name__=='__main__':unittest.main(verbosity=2)
