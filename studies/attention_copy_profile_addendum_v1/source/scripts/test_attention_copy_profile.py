"""Numerical controls, layout contract and profiler accounting tests."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import attention_copy_profile as p
from attention_copy_adapter import ProfileAttention
from review_attention_copy_profile import operator_summary
DEVICE=torch.device('cpu')


class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=p.configuration(True,DEVICE);self.spec=self.cfg['spec'];self.seed=self.spec['seeds'][0]
        x,y,_,_,_=p.c.r.v.load_data('development',DEVICE,fake=True)
        self.x,self.y,self.input=p.fixed_batch(self.seed,3,self.spec,x,y)

    def test_compatible_arm_matches_qualified_updates(self):
        oldcfg=p.c.configuration(True,DEVICE);policy=self.spec['execution_policy']
        old,opt,_=p.c.make(self.seed,policy,oldcfg,DEVICE)
        model,newopt=p.make(self.seed,'compatible','original',self.cfg,DEVICE)
        for group in opt.param_groups:group['lr']=self.spec['learning_rate']
        old.train();model.train()
        with p.c.e.policy(policy):
            for i in range(4):
                a=p.c.update(old,opt,self.x,self.y,self.seed+self.spec['dropout_seed_offset']+i,policy,self.spec)
                b=p.plain_update(model,newopt,self.x,self.y,self.seed,i,self.spec)
                self.assertEqual(a,b)
        self.assertEqual(p.state(old,opt),p.state(model,newopt))

    def test_copy_preserves_attention_values_but_changes_stride(self):
        model,_=p.make(self.seed,'compatible','original',self.cfg,DEVICE)
        module=model.body.encoder.layers[0].self_attention
        x=torch.randn(3,5,module.embed_dim,device=DEVICE)
        with p.c.e.policy(self.spec['execution_policy']),torch.autocast(DEVICE.type,dtype=torch.bfloat16,enabled=DEVICE.type=='cuda'):
            a=module(x,x,x,need_weights=False)[0]
            module.keep_copy=False;b=module(x,x,x,need_weights=False)[0]
        self.assertTrue(torch.equal(a,b));self.assertNotEqual(a.stride(),b.stride())
        self.assertEqual(a.stride(),(module.embed_dim,3*module.embed_dim,1))

    def test_dropout_condition_and_state_mapping(self):
        a,ao=p.make(self.seed,'compatible','original',self.cfg,DEVICE)
        b,bo=p.make(self.seed,'native_layout_control','disabled',self.cfg,DEVICE)
        self.assertEqual(p.state(a,ao),p.state(b,bo))
        self.assertTrue(any(m.p>0 for m in a.modules() if isinstance(m,torch.nn.Dropout)))
        self.assertTrue(all(m.p==0 for m in b.modules() if isinstance(m,torch.nn.Dropout)))
        self.assertTrue(all(t.requires_grad for t in b.parameters()))

    def test_plain_and_instrumented_update_identity(self):
        for arm in self.spec['arms']:
            a,ao=p.make(self.seed,arm,'original',self.cfg,DEVICE)
            b,bo=p.make(self.seed,arm,'original',self.cfg,DEVICE)
            rec=p.attribution.PhaseRecorder('events',DEVICE)
            with p.c.e.policy(self.spec['execution_policy']):
                for i in range(3):
                    self.assertEqual(p.plain_update(a,ao,self.x,self.y,self.seed,i,self.spec),
                                     p.instrumented_update(b,bo,self.x,self.y,self.seed,i,self.spec,rec))
            self.assertEqual(p.state(a,ao),p.state(b,bo));self.assertEqual(len(rec.rows()),15)

    def test_accounting_excludes_duplicate_kernel_events_and_missing_gpu(self):
        phase=SimpleNamespace(name='phase::forward_loss',cpu_parent=None)
        mod=SimpleNamespace(name='module::compatibility_copy_0',cpu_parent=phase)
        event=SimpleNamespace(name='aten::copy_',device_type='CPU',cpu_parent=mod,input_shapes=[[3,5,24]],self_cpu_time_total=2.,self_device_time_total=3.)
        kernel=SimpleNamespace(name='copy_kernel',device_type='CUDA')
        rows,count=p.attribution.collect_operators(SimpleNamespace(events=lambda:[event,kernel]))
        self.assertEqual(len(rows),1);self.assertEqual(count,1)
        result=operator_summary(rows,dict(gpu_operator_attribution_available=True))
        self.assertEqual(result['linked_gpu_self_us'],3.);self.assertEqual(result['compatibility_gpu_share_percent'],100.)
        result=operator_summary(rows,dict(gpu_operator_attribution_available=False))
        self.assertIsNone(result['linked_gpu_self_us']);self.assertIsNone(result['compatibility_gpu_share_percent'])

    def test_case_coverage_and_arm_orders(self):
        spec=p.read(p.SPEC);base=set(p.cases(spec,0));self.assertEqual(len(base),8)
        for s in range(4):self.assertEqual(set(p.cases(spec,s)),base)
        for case in base:
            orders=[p.arm_order(spec,s,p.cases(spec,s).index(case))[0] for s in range(4)]
            self.assertEqual(orders.count('compatible'),2)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path);args=ap.parse_args()
    DEVICE=torch.device(args.device);p.c.r.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        p.save(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
