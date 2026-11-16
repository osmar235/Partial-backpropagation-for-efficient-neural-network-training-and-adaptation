"""Routing correctness, frozen-state behavior and balanced costing controls."""
import argparse
import copy
import unittest
from collections import Counter
import torch
import optimized_prefix_cost as p
from review_optimized_prefix_cost import pair_cost

DEVICE = torch.device('cpu')


class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = p.configuration(True, DEVICE); self.spec = self.cfg['spec']
        self.policy = self.spec['arms'][0]; self.seed = self.spec['seeds'][0]
        self.x, self.y, _, _, _ = p.c.r.v.load_data('development', DEVICE, fake=True)

    def model(self):
        model, opt, _ = p.c.make(self.seed, self.policy, self.cfg, DEVICE)
        model.train(); return model, opt

    def test_partition_forward_and_suffix_gradients(self):
        model, opt = self.model(); reference = copy.deepcopy(model)
        x = p.c.r.v.frozen6.normalize_cifar(self.x[:3]); targets = self.y[:3]
        with p.c.e.policy(self.policy):
            torch.manual_seed(891); logits, loss = p.c.forward_loss(reference, x, targets, self.policy, self.spec)
            loss.backward()
            for boundary in self.spec['boundaries']:
                model.requires_grad_(True); model.zero_grad(set_to_none=True)
                router = p.Router(model, boundary); router.select(False)
                torch.manual_seed(891); actual, candidate = p.c.forward_loss(model, x, targets, self.policy, self.spec)
                self.assertTrue(torch.equal(actual, logits)); candidate.backward()
                refs = dict(reference.named_parameters())
                for name, param in model.named_parameters():
                    if name in router.prefix: self.assertIsNone(param.grad)
                    else: self.assertTrue(torch.equal(param.grad, refs[name].grad), name)
                self.assertIn('body.class_token', router.prefix)
                self.assertIn('body.encoder.pos_embedding', router.prefix)
                self.assertNotIn('body.encoder.ln.weight', router.prefix)
                self.assertNotIn('head.weight', router.prefix)

    def test_stale_gradients_decay_and_ages(self):
        anchor = p.make_anchor(self.seed, self.cfg, DEVICE, self.x, self.y)
        model, opt = self.model(); p.restore(model, opt, anchor)
        arm = next(a for a in p.arms(self.spec) if a['name'] == 'cut1_r4')
        router = p.Router(model, 1); stream = p.Stream(self.seed, self.spec, self.x, self.y)
        with p.c.e.policy(self.policy):
            p.step(model, opt, router, arm, 0, self.seed, stream, self.policy, self.spec)
            weights = {n: t.detach().clone() for n, t in router.prefix.items()}
            moments = {n: copy.deepcopy(opt.state[t]) for n, t in router.prefix.items()}
            for i in range(1, 4): p.step(model, opt, router, arm, i, self.seed, stream, self.policy, self.spec)
        for name, param in router.prefix.items():
            self.assertIsNone(param.grad); self.assertTrue(torch.equal(param, weights[name]))
            for key, val in moments[name].items(): self.assertTrue(torch.equal(opt.state[param][key], val))
        with p.c.e.policy(self.policy):
            for i in range(4, self.spec['updates']): p.step(model, opt, router, arm, i, self.seed, stream, self.policy, self.spec)
        self.assertEqual(p.state_check(model, opt, router, arm, anchor, self.spec)['status'], 'PASS')

    def test_restoration_and_stream_replay(self):
        anchor = p.make_anchor(self.seed, self.cfg, DEVICE, self.x, self.y)
        model, opt = self.model(); p.restore(model, opt, anchor)
        s1 = p.Stream(self.seed, self.spec, self.x, self.y); s2 = p.Stream(self.seed, self.spec, self.x, self.y)
        for _ in range(9):
            a, b = s1.batch(); x, y = s2.batch()
            self.assertTrue(torch.equal(a, x)); self.assertTrue(torch.equal(b, y))
        self.assertEqual(s1.fingerprint(), s2.fingerprint())
        arm = p.arms(self.spec)[2]; router = p.Router(model, arm['boundary'])
        with p.c.e.policy(self.policy): p.step(model, opt, router, arm, 0, self.seed, s1, self.policy, self.spec)
        p.restore(model, opt, anchor)
        self.assertEqual(p.c.r.old.state_hash(model.state_dict()), anchor['model_hash'])
        self.assertEqual(p.c.r.optimizer_hash(opt), anchor['optimizer_hash'])
        self.assertTrue(all(t.requires_grad and t.grad is None for t in model.parameters()))

    def test_schedule_refresh_budget_and_cost_sign(self):
        spec = p.read(p.SPEC); choices = p.arms(spec); n = len(choices)
        self.assertEqual(n, 14)
        for seed in spec['seeds']:
            orders = [[a['name'] for s, a in p.schedule(spec, session) if s == seed] for session in range(4)]
            for arm in choices:
                self.assertEqual(sum(order.index(arm['name']) for order in orders), 2*(n-1))
                self.assertEqual(sum(p.full_step(arm, i) for i in range(96)), 0 if arm['interval'] == 0 else 96//arm['interval'])
        self.assertEqual(pair_cost(10, 8), 100*(1-8/10))
        self.assertLess(pair_cost(10, 11), 0)
        with self.assertRaises(AssertionError): pair_cost(0, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--output', type=p.Path); args = ap.parse_args()
    DEVICE = torch.device(args.device); p.c.r.m.configure()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        assert not args.output.exists(); args.output.parent.mkdir(parents=True, exist_ok=True)
        p.save(args.output, dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun, device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
