"""Check inference, pairing, and prospective scheduling; no GPU timing claims."""
import copy
import itertools
import unittest
from review_qkv_confirmation import statistics,same_trajectory
import qkv_confirmation as c

class Tests(unittest.TestCase):
    def test_exact_sign_and_family_adjustment(self):
        s=statistics([5.]*8,4)
        self.assertEqual(s['one_sided_sign_p'],1/256)
        self.assertEqual(s['bonferroni_p'],.015625)
        self.assertTrue(s['directional_pass'])
        self.assertFalse(statistics([5.]*7+[-.1],4)['directional_pass'])
        self.assertFalse(statistics([5.]*7+[0.],4)['directional_pass'])
        self.assertFalse(statistics([-1.]*8,4)['directional_pass'])

    def test_interval_coverage_and_units(self):
        s=statistics([1.,2.,3.,4.,5.,6.,7.,8.],4)
        self.assertEqual(s['median_saving_percent'],4.5)
        self.assertEqual(s['median_interval_percent'],[1.,8.])
        self.assertEqual(s['marginal_interval_coverage'],.9921875)
        self.assertEqual(s['simultaneous_interval_coverage_lower_bound'],.96875)
        signs=list(itertools.product((-1,1),repeat=8))
        self.assertEqual(sum(statistics(list(z),4)['directional_pass'] for z in signs),1)

    def test_balanced_condition_and_arm_order(self):
        spec=c.read(c.SPEC);n=len(spec['seeds'])*len(spec['batch_sizes'])
        conditions=list(itertools.product(spec['seeds'],spec['batch_sizes']))
        for condition in conditions:
            positions=[c.schedule(spec,s).index(condition) for s in range(spec['sessions'])]
            self.assertEqual([positions.count(p) for p in range(n)],[2]*n)
        for s in range(spec['sessions']):
            for pos in range(n):
                orders=[c.e.order(spec['arms'],pos+s,r)[0] for r in range(spec['rounds'])]
                self.assertEqual(orders.count('math'),2)
                self.assertEqual(orders.count('qkv_unbind'),2)

    def test_trajectory_rejects_any_changed_component(self):
        row=dict(first_loss=2.,steady_losses=[1.,.8],final_model_sha256='m',final_optimizer_sha256='o',final_rng={'cpu':'x','cuda':'y'})
        self.assertTrue(same_trajectory(row,copy.deepcopy(row)))
        for key in row:
            changed=copy.deepcopy(row);changed[key]=None
            self.assertFalse(same_trajectory(row,changed))

    def test_preserved_candidate_and_tolerances(self):
        spec=c.read(c.SPEC);previous=c.read(c.e.SPEC)
        for k in ('arms','tolerances','learning_rate','weight_decay','label_smoothing'):
            self.assertEqual(spec[k],previous[k])
        self.assertTrue(set(spec['seeds']).isdisjoint(previous['seeds']))
        self.assertEqual(spec['sessions']*len(spec['seeds'])*len(spec['batch_sizes'])*spec['rounds']*2*(1+spec['steady_updates']),33024)

if __name__=='__main__':unittest.main(verbosity=2)
