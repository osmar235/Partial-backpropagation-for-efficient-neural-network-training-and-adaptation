"""Mechanics, causal allocation, exact budgets and target-accounting tests."""
import copy, itertools, math, random, unittest
import torch
import stage8_vit_v2 as v
import stage9_quality_pilot as p
from review_stage9E import target

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);v.configure()

    def test_continuation_counts_against_enumeration(self):
        for n in range(9):
            for previous in [False,True]:
                for k in range(n+1):
                    valid=0
                    for mask in itertools.product([False,True],repeat=n):
                        if sum(mask)!=k:continue
                        extended=(previous,)+mask
                        valid+=all(not (a and b) for a,b in zip(extended,extended[1:]))
                    self.assertEqual(p.ways(n,k,previous),valid,(n,k,previous))

    def test_online_quotas_under_adversarial_requests(self):
        spec=dict(quota_window=20,omission_fraction=.4)
        for strategy in ['periodic','random','mixup_only','boundary_rms']:
            for request_type in ['always','never','alternating']:
                budget=p.Budget(spec);rng=random.Random(123)
                for _ in range(3):
                    budget.begin_epoch(175);previous=False;total=0
                    for i in range(175):
                        request=request_type=='always' or (request_type=='alternating' and i%2==0)
                        choice,*_=budget.choose(strategy,request,rng.random())
                        self.assertFalse(previous and choice);total+=choice;previous=choice
                    budget.end_epoch();self.assertEqual(total,70)
                    budget.mandatory_full() # The common final short batch.

    def test_current_feature_does_not_enter_its_threshold(self):
        spec=dict(history_size=4,calibration_quantile=.4,quota_window=20,omission_fraction=.4)
        for variant in ['mixup_only','boundary_rms']:
            a=p.Controller(variant,1301,spec);a.history.extend([1.,2.,3.,4.]);a.budget.begin_epoch(20)
            _,metadata=a.decide(-100.,True,True)
            self.assertAlmostEqual(metadata['threshold'],2.2)
            self.assertEqual(list(a.history),[1.,2.,3.,4.])
            a.observe(-100.);self.assertEqual(list(a.history),[2.,3.,4.,-100.])

    def test_native_and_tiny_kernel_equivalence_and_freeze_state(self):
        for tiny,cut in [(True,1),(False,4)]:
            v.seed_all(11901);model=v.make_model(tiny=tiny)
            torch.nn.init.normal_(model.heads.head.weight,std=.02)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
            x=torch.rand(3,3,32,32);labels=torch.tensor([0,1,2])
            targets=torch.nn.functional.one_hot(labels,10).float()
            for _ in range(2):p.kernel(model,opt,x,targets,cut,lambda signal:(False,{}))
            weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict())
            _,_,prefix_names=v.partition(model,cut);snapshots={}
            for mode in ['ordinary_full','predecided_freeze','boundary_full','boundary_freeze']:
                work=copy.deepcopy(model);adam=torch.optim.AdamW(work.parameters(),lr=.001,weight_decay=.05);adam.load_state_dict(copy.deepcopy(state))
                v.seed_all(11902)
                loss,partial,signal,_=p.kernel(work,adam,x,targets,cut,lambda value:(mode.endswith('freeze'),{}),boundary=mode.startswith('boundary'))
                snapshots[mode]=(work,adam,loss)
                if partial:
                    baseline_by_name=dict(model.named_parameters())
                    for name,param in work.named_parameters():
                        if name in prefix_names:
                            self.assertIsNone(param.grad);self.assertTrue(torch.equal(param,weights[name]))
                            for field in ['step','exp_avg','exp_avg_sq']:
                                self.assertTrue(torch.equal(adam.state[param][field],opt.state[baseline_by_name[name]][field]))
                self.assertEqual(signal is not None,mode.startswith('boundary'))
            for a,b in [('ordinary_full','boundary_full'),('predecided_freeze','boundary_freeze')]:
                wa,oa,la=snapshots[a];wb,ob,lb=snapshots[b];self.assertAlmostEqual(la,lb,places=6)
                for (na,pa),(nb,pb) in zip(wa.named_parameters(),wb.named_parameters()):
                    self.assertEqual(na,nb);torch.testing.assert_close(pa,pb,rtol=1e-4,atol=1e-6)
                    if pa.grad is None or pb.grad is None:self.assertIsNone(pa.grad);self.assertIsNone(pb.grad)
                    else:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-6)
                    for field in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(oa.state[pa][field],ob.state[pb][field],rtol=1e-4,atol=1e-6)
            ordinary=snapshots['ordinary_full'][0]
            self.assertTrue(any(float(param.grad.norm())>0 for name,param in ordinary.named_parameters() if name in prefix_names))

    def test_sustained_target_and_censoring(self):
        rows=[dict(epoch=i+1,validation_accuracy=a) for i,a in enumerate([.79,.80,.78,.81,.82,.83])]
        first,confirmed=target(rows,.8,3)
        self.assertEqual(first['epoch'],2);self.assertEqual(confirmed['epoch'],6)
        first,confirmed=target(rows,.9,3);self.assertIsNone(first);self.assertIsNone(confirmed)
        first,confirmed=target(rows,.82,3);self.assertEqual(first['epoch'],5);self.assertIsNone(confirmed)

if __name__=='__main__':unittest.main(verbosity=2)
