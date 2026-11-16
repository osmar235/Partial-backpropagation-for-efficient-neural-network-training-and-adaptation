"""Nontrivial function/derivative preservation, reset semantics and trainability."""
import copy
import sys
import unittest
import torch
import torch.nn.functional as F
import stage10_models as m
import stage11_growth as g
from stage10_representation import update

DEVICE = 'cuda' if '--cuda' in sys.argv else 'cpu'
if '--cuda' in sys.argv:
    sys.argv.remove('--cuda')


class GrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m.configure()
        if DEVICE == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')

    def model(self, fake=True, dtype=torch.float32):
        model = m.make_model('small_rgb32', 1501, fake).to(device=DEVICE, dtype=dtype)
        # A zero classifier would make a broken growth operator pass trivially.
        torch.manual_seed(150109)
        with torch.no_grad():
            model.head.weight.normal_(std=.1)
            model.head.bias.normal_(std=.02)
            for name, p in model.named_parameters():
                if name.endswith('bias') and not name.startswith('head'):
                    p.normal_(std=.01)
        return model

    def test_native_function_and_input_derivative(self):
        dtype = torch.float64 if DEVICE == 'cpu' else torch.float32
        small = self.model(fake=False, dtype=dtype).eval()
        before = {n: p.clone() for n, p in small.state_dict().items()}
        large = g.expand(small, 1501, False)
        if DEVICE == 'cuda':
            # The read-only auditor reconstructs this map on CPU from GPU saves.
            cpu_map = g.lift_state({n:t.cpu() for n,t in small.state_dict().items()})
            for name,tensor in large.state_dict().items():
                self.assertTrue(torch.equal(tensor.cpu(),cpu_map[name]),name)
        torch.manual_seed(150110)
        x = torch.randn(2, 3, 32, 32, device=DEVICE, dtype=dtype, requires_grad=True)
        z = x.detach().clone().requires_grad_()
        a, b = small(x), large(z)
        tol = 1e-9 if dtype == torch.float64 else 5e-5
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad, z.grad, rtol=tol * 10, atol=tol)
        self.assertEqual(sum(p.numel() for p in large.parameters()), 2693578)
        self.assertEqual(large.body.encoder.layers[0].self_attention.num_heads, 3)
        for n, p in small.state_dict().items():
            self.assertTrue(torch.equal(p, before[n]), n)

    def test_parameter_gradient_pullback(self):
        dtype = torch.float64 if DEVICE == 'cpu' else torch.float32
        small = self.model(dtype=dtype)
        large = g.expand(small, 1501, True)
        g.disable_dropout(small); g.disable_dropout(large)
        x = torch.randn(3, 3, 32, 32, device=DEVICE, dtype=dtype)
        y = torch.tensor([1, 4, 9], device=DEVICE)
        F.cross_entropy(small(x), y).backward()
        F.cross_entropy(large(x), y).backward()
        params = dict(small.named_parameters())
        mapped = g.lift_state(params)
        dual = sum((mapped[n] * p.grad.detach()).sum() for n, p in large.named_parameters())
        pullback = torch.autograd.grad(dual, tuple(params.values()))
        tol = 1e-8 if dtype == torch.float64 else 2e-4
        for (n, p), actual in zip(params.items(), pullback):
            torch.testing.assert_close(actual, p.grad, rtol=tol, atol=tol, msg=n)

    def test_wrong_attention_scaling_is_detectable(self):
        small = self.model().eval(); large = g.expand(small, 1501, True)
        x = torch.randn(3, 3, 32, 32, device=DEVICE)
        with torch.no_grad():
            for layer in large.body.encoder.layers:
                n = layer.self_attention.embed_dim
                layer.self_attention.in_proj_weight[:2 * n].mul_(2 ** .25)
                layer.self_attention.in_proj_bias[:2 * n].mul_(2 ** .25)
            error = float((small(x) - large(x)).abs().max())
        self.assertGreater(error, 1e-4)

    def test_growth_reset_learning_and_replay(self):
        small = self.model(); opt = g.reset_adam(small)
        x = torch.randn(3, 3, 32, 32, device=DEVICE)
        targets = F.one_hot(torch.tensor([0, 4, 8], device=DEVICE), 10).float()
        for i in range(3):
            update(small, opt, x, targets, 150201 + i)
        self.assertTrue(all(int(s['step']) == 3 for s in opt.state.values()))
        snapshots = []
        for _ in range(2):
            large = g.expand(small, 1501, True)
            self.assertEqual(g.twin_difference(large), 0.)
            new_opt = g.reset_adam(large)
            self.assertEqual(len(new_opt.state), 0)
            losses = [update(large, new_opt, x, targets, 150301 + i) for i in range(5)]
            self.assertGreater(g.twin_difference(large), 1e-8)
            for p in large.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertEqual(int(new_opt.state[p]['step']), 5)
            snapshots.append((losses, copy.deepcopy(large.state_dict()), copy.deepcopy(new_opt.state_dict())))
        self.assertEqual(snapshots[0][0], snapshots[1][0])
        for n in snapshots[0][1]:
            self.assertTrue(torch.equal(snapshots[0][1][n], snapshots[1][1][n]), n)
        for key, state in snapshots[0][2]['state'].items():
            for n, value in state.items():
                self.assertTrue(torch.equal(value, snapshots[1][2]['state'][key][n]))

    def test_unknown_parameters_rejected(self):
        state = self.model().state_dict()
        state['unsupported.weight'] = torch.ones(1, device=DEVICE)
        with self.assertRaises(ValueError):
            g.lift_state(state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
