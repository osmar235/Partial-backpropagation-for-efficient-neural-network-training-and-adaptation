"""Artificial invariants; no real data or source checkpoint scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import unittest
import torch
import torch.nn.functional as F
import stage10_models as m
from stage14_tokens import TokenModel, merge_tokens, proportional_block
from review_stage14A import screen

DEVICE = 'cpu'


def core(fake=True):
    net = m.make_model('full_rgb32', 9871, fake).to(DEVICE)
    with torch.no_grad():
        torch.manual_seed(123)
        net.head.weight.normal_(std=.05)
    return net


class Tests(unittest.TestCase):
    def test_full_identity_and_gradients(self):
        a = core(); b = copy.deepcopy(a)
        a.eval(); b.eval()
        x = torch.randn(3, 3, 32, 32, device=DEVICE)
        za = a(x)
        # Independent manual token path with unit masses must reproduce native
        # logits AND parameter gradients before any compression is introduced.
        tokens = b.body._process_input(x)
        tokens = torch.cat((b.body.class_token.expand(len(x), -1, -1), tokens), 1)
        tokens = b.body.encoder.dropout(tokens + b.body.encoder.pos_embedding)
        mass = torch.ones(3, 65, device=DEVICE)
        for block in b.body.encoder.layers:
            tokens = proportional_block(block, tokens, mass)
        zb = b.head(b.body.encoder.ln(tokens)[:, 0])
        torch.testing.assert_close(za, zb, rtol=2e-5, atol=2e-5)
        za.square().sum().backward(); zb.square().sum().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-5)

    def test_mass_and_every_token_gradient(self):
        for mode in ('spatial', 'similarity'):
            x = torch.randn(3, 65, 24, device=DEVICE, requires_grad=True)
            z, sizes = merge_tokens(x, mode)
            assert z.shape == (3, 33, 24)
            torch.testing.assert_close(z[:, 0], x[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(sizes.sum(1), torch.full((3,), 65., device=DEVICE))
            torch.testing.assert_close((z * sizes[..., None]).sum(1), x.sum(1), rtol=1e-5, atol=1e-5)
            z.sum().backward()
            assert torch.all(x.grad > 0) and torch.isfinite(x.grad).all()
            if mode == 'spatial':
                torch.testing.assert_close(z[:, 1:], (x[:, 1::2] + x[:, 2::2]) / 2)
                assert torch.all(sizes[:, 1:] == 2)

    def test_proportional_duplicate_equivalence(self):
        block = core().body.encoder.layers[0].eval()
        # Repeated identical patch features: merging must preserve retained
        # token outputs, including CLS, after both attention and per-token MLP.
        small = torch.randn(2, 33, 24, device=DEVICE)
        large = torch.cat((small[:, :1], small[:, 1:].repeat_interleave(2, dim=1)), 1)
        mass = torch.full((2, 33), 2., device=DEVICE); mass[:, 0] = 1
        with torch.no_grad():
            expected = block(large)
            actual = proportional_block(block, small, mass)
            torch.testing.assert_close(actual[:, :1], expected[:, :1], rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(actual[:, 1:], expected[:, 2::2], rtol=2e-5, atol=2e-5)
            ones = torch.ones(2, 33, device=DEVICE)
            torch.testing.assert_close(proportional_block(block, small, ones), block(small), rtol=2e-5, atol=2e-5)

    def test_native_compressed_backprop_and_state(self):
        net = core(fake=False)
        before = {n: t.clone() for n, t in net.state_dict().items()}
        x = torch.randn(2, 3, 32, 32, device=DEVICE)
        for mode in ('spatial', 'similarity'):
            wrap = TokenModel(net, mode, 3).train()
            wrap.zero_grad(set_to_none=True)
            F.cross_entropy(wrap(x), torch.tensor([1, 3], device=DEVICE)).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            assert net.body.conv_proj.weight.grad.norm() > 0
            assert (net.body.encoder.pos_embedding.grad.abs().sum(-1) > 0).all()
            for n, t in net.state_dict().items():
                torch.testing.assert_close(t, before[n], rtol=0, atol=0)

    def test_screen_boundaries(self):
        spec = dict(accuracy_margin_pp=.3, minimum_time_saving=.1, minimum_incremental_memory_saving=.2)
        assert screen(.797, .8, .9, .99, spec, True, False)['seed_screen_pass']
        assert screen(.8, .8, 1.1, .8, spec, True, False)['seed_screen_pass']
        assert not screen(.7969, .8, .8, .7, spec, True, False)['seed_screen_pass']
        assert not screen(.8, .8, .901, .801, spec, True, False)['seed_screen_pass']
        assert not screen(.8, .8, .8, .7, spec, False, False)['seed_screen_pass']
        assert not screen(.8, .8, .8, .7, spec, True, True)['seed_screen_pass']


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()
    DEVICE = 'cuda' if args.cuda else 'cpu'
    m.configure()
    unittest.main(argv=['test_stage14_tokens'], verbosity=2)
