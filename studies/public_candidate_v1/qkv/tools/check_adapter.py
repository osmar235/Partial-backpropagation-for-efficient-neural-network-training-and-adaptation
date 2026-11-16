"""Portable small correctness check of the unchanged study adapter, no timing claim."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qkv_unbind import UnbindSelfAttention


def check(device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(99151)
    native = nn.MultiheadAttention(12, 3, dropout=0, batch_first=True).to(device=device, dtype=torch.float64)
    original = copy.deepcopy(native)
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-4)
    identities = {name: id(p) for name, p in original.named_parameters()}
    changed = UnbindSelfAttention(original)
    assert identities == {name: id(p) for name, p in changed.named_parameters()}
    assert set(native.state_dict()) == set(changed.state_dict())
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == set(identities.values())
    x = torch.randn(2, 7, 12, device=device, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    upstream = torch.randn_like(x)
    with sdpa_kernel(SDPBackend.MATH):
        a = native(x, x, x, need_weights=False)[0]
        b = changed(xx, xx, xx, need_weights=False)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.backward(upstream)
        b.backward(upstream)
    torch.testing.assert_close(x.grad, xx.grad, atol=1e-12, rtol=1e-12)
    for name, param in native.named_parameters():
        torch.testing.assert_close(param.grad, dict(changed.named_parameters())[name].grad, atol=1e-12, rtol=1e-12)
    for kwargs in ({'need_weights': True}, {'need_weights': False, 'is_causal': True}):
        try:
            changed(xx, xx, xx, **kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unsupported input was accepted')
    return dict(status='PASS', device=device, torch=str(torch.__version__),
                checks=['exact outputs', 'all parameter and input gradients within 1e-12',
                        'parameter/optimizer identity', 'checkpoint keys', 'unsupported inputs rejected'],
                training_rerun=False, performance_measured=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    print(json.dumps(check(parser.parse_args().device), indent=2))
