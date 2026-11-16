"""Controls must isolate growth/reset; outcome screen must respect quality."""
import copy,json,sys,unittest
import torch
import torch.nn.functional as F
import stage11_growth_pilot as p
from review_stage11B import opt_equal

DEVICE='cuda' if '--cuda' in sys.argv else 'cpu'
if '--cuda' in sys.argv:sys.argv.remove('--cuda')

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.m.configure();cls.spec=json.loads(p.SPEC.read_text())

    def test_all_transition_controls_and_following_update(self):
        torch.manual_seed(921)
        x=torch.randn(3,3,32,32,device=DEVICE)
        y=F.one_hot(torch.tensor([0,4,8],device=DEVICE),10).float()
        for arm in ['full','small','full_reset','small_reset','grow']:
            with self.subTest(arm=arm):
                model=p.m.make_model(p.initial_variant(arm),9971,True).to(DEVICE)
                opt=p.g.reset_adam(model)
                for i in range(3):p.old.update(model,opt,x,y,931+i)
                state=copy.deepcopy(model.state_dict());moments=copy.deepcopy(opt.state_dict())
                opt.param_groups[0]['lr']=.0007;moments['param_groups'][0]['lr']=.0007
                model,opt,info=p.transition(model,opt,arm,9971,True,self.spec,x)
                self.assertEqual(opt.param_groups[0]['lr'],.0007)
                self.assertLessEqual(info['max_scaled_logit_error'],1.)
                self.assertEqual(info['prediction_changes'],0)
                expected=p.g.lift_state(state) if arm=='grow' else state
                for n,t in expected.items():self.assertTrue(torch.equal(t,model.state_dict()[n]),n)
                if p.resets(arm):self.assertFalse(opt.state)
                else:opt_equal(moments,opt.state_dict())
                p.old.update(model,opt,x,y,935)
                self.assertTrue(all(int(s['step'])==(1 if p.resets(arm) else 4) for s in opt.state.values()))
                self.assertTrue(all(torch.isfinite(t).all() for t in model.parameters()))

    def records(self):
        return {k:dict(final_validation_accuracy=a,total_run_seconds=t,readiness_reached=True)
                for k,a,t in [('full',.82,100),('small',.80,60),('full_reset',.819,101),('small_reset',.801,61),('grow',.819,90)]}

    def test_low_quality_small_is_not_qualified(self):
        r=p.practical_screen(self.records(),self.spec)
        self.assertTrue(r['seed_screen_pass']);self.assertEqual(r['fastest_qualified_control'],'full')

    def test_competitive_small_blocks_slower_growth(self):
        records=self.records();records['small']['final_validation_accuracy']=.819
        r=p.practical_screen(records,self.spec)
        self.assertTrue(r['quality_pass']);self.assertFalse(r['speed_pass'])
        self.assertEqual(r['fastest_qualified_control'],'small')

    def test_reset_control_quality_and_readiness_cannot_be_ignored(self):
        records=self.records();records['small_reset']['final_validation_accuracy']=.83
        self.assertFalse(p.practical_screen(records,self.spec)['quality_pass'])
        for arm in ['full','grow']:
            records=self.records();records[arm]['readiness_reached']=False
            self.assertFalse(p.practical_screen(records,self.spec)['seed_screen_pass'])

    def test_margin_and_time_boundaries(self):
        records=self.records();records['grow'].update(final_validation_accuracy=.818,total_run_seconds=95.)
        self.assertTrue(p.practical_screen(records,self.spec)['seed_screen_pass'])
        records['grow']['total_run_seconds']=95.01
        self.assertFalse(p.practical_screen(records,self.spec)['speed_pass'])
        records['grow'].update(final_validation_accuracy=.8179,total_run_seconds=90.)
        self.assertFalse(p.practical_screen(records,self.spec)['quality_pass'])

if __name__=='__main__':unittest.main(verbosity=2)
