"""Artificial checkpoint continuation, reset isolation and qualification checks."""
import copy
import sys
import unittest
import torch
import stage13b_reference as r
from review_stage13B import qualifies

DEVICE = torch.device('cuda' if '--cuda' in sys.argv else 'cpu')
if '--cuda' in sys.argv:
    sys.argv.remove('--cuda')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.p.execution.configure()
        cls.recipe = dict(__import__('json').loads(r.prior.SPEC.read_text()), image_size=32,
            batch_size=3, eval_batch_size=4, batches_per_epoch=6)
        cls.data = r.p.load_data(DEVICE, True)

    def test_saved_continuation_matches_uninterrupted(self):
        cp, initial, expected, trace = r.fake_source(9971, self.data, self.recipe, DEVICE)
        net, opt, bg, ag = r.restore(9971, cp, self.recipe, True, DEVICE, 'continuation')
        torch.testing.assert_close(torch.from_numpy(r.predict(net, self.data[2], self.recipe)), torch.from_numpy(initial), atol=0, rtol=0)
        for j, ix in enumerate(r.v.frozen6.epoch_batches(16, 3, bg)):
            ix = ix.to(DEVICE); xx = r.p.transform(self.data[0][ix], 32, ag)
            r.rates(opt, cp['updates'] + j, self.recipe)
            loss = r.p.update(net, opt, xx, self.data[1][ix], 'full', 0.)
            self.assertEqual(loss, trace[j]['loss'])
        torch.testing.assert_close(torch.from_numpy(r.predict(net, self.data[2], self.recipe)), torch.from_numpy(expected), atol=0, rtol=0)

    def test_reset_changes_only_optimizer_history(self):
        cp, _, _, _ = r.fake_source(9971, self.data, self.recipe, DEVICE)
        a, ao, bg, ag = r.restore(9971, cp, self.recipe, True, DEVICE, 'continuation')
        b, bo, bbg, bag = r.restore(9971, cp, self.recipe, True, DEVICE, 'reset')
        self.assertEqual(r.old.state_hash(a.state_dict()), r.old.state_hash(b.state_dict()))
        self.assertTrue(torch.equal(bg.get_state(), bbg.get_state()) and torch.equal(ag.get_state(), bag.get_state()))
        self.assertTrue(all(age == (12 if name.startswith('fc.') else 0) for name, age in r.ages(a, ao).items()))
        self.assertTrue(all(age == 0 for age in r.ages(b, bo).values()))
        self.assertEqual(r.rates(ao, 12, self.recipe), r.rates(bo, 12, self.recipe))
        xx = r.p.transform(self.data[0][:3], 32)
        buffers = {n: t.clone() for n, t in a.named_buffers()}
        r.p.update(a, ao, xx, self.data[1][:3], 'full', 0.)
        r.p.update(b, bo, xx, self.data[1][:3], 'full', 0.)
        self.assertFalse(torch.equal(a.fc.weight, b.fc.weight))
        for name, t in a.named_parameters():
            if not name.startswith('fc.'):
                torch.testing.assert_close(t, dict(b.named_parameters())[name], atol=0, rtol=0)
        for name, t in a.named_buffers():
            torch.testing.assert_close(t, buffers[name], atol=0, rtol=0)

    def test_rng_consumption_audit(self):
        a = torch.Generator().manual_seed(45); b = torch.Generator().manual_seed(45)
        for size in (3, 3, 1):
            r.p.transform(self.data[0][:size], 32, a)
            torch.randint(0, 9, (size, 2), generator=b); torch.rand(size, generator=b)
        self.assertTrue(torch.equal(a.get_state(), b.get_state()))

    def test_native_checkpoint_restore(self):
        # Artificial inputs; native model, classifier moments and absent backbone moments.
        net = r.p.make_model(9971, False).to(DEVICE); opt = r.p.optimizer(net, self.recipe)
        xx = r.p.transform(self.data[0][:2], 96)
        r.p.update(net, opt, xx, self.data[1][:2], 'head', 0.)
        cp = copy.deepcopy(dict(model_state=net.state_dict(), optimizer_state=opt.state_dict(),
            batch_rng=torch.Generator().get_state(), augmentation_rng=torch.Generator().get_state()))
        restored, ro, _, _ = r.restore(9971, cp, self.recipe, False, DEVICE, 'continuation')
        r.rates(opt, 704, self.recipe); r.rates(ro, 704, self.recipe)
        r.p.update(net, opt, xx, self.data[1][:2], 'full', 0.)
        r.p.update(restored, ro, xx, self.data[1][:2], 'full', 0.)
        self.assertEqual(r.old.state_hash(net.state_dict()), r.old.state_hash(restored.state_dict()))
        self.assertEqual(r.ages(net, opt), r.ages(restored, ro))

    def test_qualification_requires_reproduction_and_improvement(self):
        initial = dict(loss=.55, accuracy=.81); final = dict(loss=.3, accuracy=.89)
        self.assertTrue(qualifies(True, initial, final, False))
        self.assertFalse(qualifies(False, initial, final, False))
        self.assertFalse(qualifies(True, initial, final, True))
        self.assertFalse(qualifies(True, initial, dict(loss=.56, accuracy=.89), False))
        self.assertFalse(qualifies(True, initial, dict(loss=.3, accuracy=.8), False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
