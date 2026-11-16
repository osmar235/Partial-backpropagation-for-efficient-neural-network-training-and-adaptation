from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

import agrls_stage1_mnist_fullbatch_v081 as v081


class SmallFashionCNN(nn.Module):
    """Same deterministic CNN used in Stage 3."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_batch_stream(n: int, batch_size: int, seed: int) -> Iterator[torch.Tensor]:
    """Infinite deterministic shuffled full-size mini-batch stream.

    The incomplete tail is dropped so all timed NN operations use the same batch
    size and the pure-compute budget remains well defined.
    """
    if batch_size <= 0 or batch_size > n:
        raise ValueError('batch_size must be in [1, n]')
    gen = torch.Generator(device='cpu').manual_seed(seed + 91001)
    while True:
        perm = torch.randperm(n, generator=gen)
        full = (n // batch_size) * batch_size
        for start in range(0, full, batch_size):
            yield perm[start:start + batch_size]


def batch_from_indices(x: torch.Tensor, y: torch.Tensor, idx_cpu: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = idx_cpu.to(device=device, non_blocking=False)
    return x.index_select(0, idx), y.index_select(0, idx)


def calibrate_pure_batch_costs(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    warmup: int = 4,
    reps: int = 15,
) -> dict:
    model.load_state_dict(init_state)
    model.to(device)
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        v081.sync(device)

    f_times: List[float] = []
    b_times: List[float] = []
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        v081.sync(device)
        t0 = time.perf_counter()
        loss = loss_fn(model(xb), yb)
        v081.sync(device)
        f_times.append(time.perf_counter() - t0)

        v081.sync(device)
        t0 = time.perf_counter()
        loss.backward()
        v081.sync(device)
        b_times.append(time.perf_counter() - t0)

    fmed = float(np.median(f_times))
    bmed = float(np.median(b_times))
    return {
        'pure_forward_median_s': fmed,
        'pure_backward_median_s': bmed,
        'pure_full_gradient_forward_ratio': (fmed + bmed) / max(fmed, 1e-15),
    }


def timed_forward_value(
    model: nn.Module,
    theta: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Copy theta outside timer, then time only model+loss forward."""
    v081.copy_vector_to_params(model, theta)
    model.train()
    model.zero_grad(set_to_none=True)
    v081.sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        loss = loss_fn(model(xb), yb)
    v081.sync(device)
    dt = time.perf_counter() - t0
    return float(loss.item()), dt


def timed_value_grad(
    model: nn.Module,
    theta: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor, float, float]:
    """Time pure forward and pure backward separately; vector plumbing is wall overhead."""
    v081.copy_vector_to_params(model, theta)
    model.train()
    model.zero_grad(set_to_none=True)

    v081.sync(device)
    t0 = time.perf_counter()
    loss = loss_fn(model(xb), yb)
    v081.sync(device)
    fsec = time.perf_counter() - t0

    v081.sync(device)
    t0 = time.perf_counter()
    loss.backward()
    v081.sync(device)
    bsec = time.perf_counter() - t0
    g = v081.grads_to_vector(model).detach().clone()
    return float(loss.detach().item()), g, fsec, bsec


def run_adam_minibatch(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    updates: int,
    batch_size: int,
    seed: int,
    lr: float,
    pure_f: float,
    pure_b: float,
) -> Tuple[List[dict], dict]:
    model.load_state_dict(init_state)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stream = make_batch_stream(int(xtr.shape[0]), batch_size, seed)

    rows: List[dict] = []
    fsec = bsec = osec = 0.0
    fpasses = bpasses = 0
    examples_forwarded = examples_backwarded = 0

    v081.sync(device)
    wall_t0 = time.perf_counter()
    for step in range(1, updates + 1):
        idx = next(stream)
        xb, yb = batch_from_indices(xtr, ytr, idx, device)
        opt.zero_grad(set_to_none=True)

        v081.sync(device)
        t0 = time.perf_counter()
        model.train()
        loss = loss_fn(model(xb), yb)
        v081.sync(device)
        fsec += time.perf_counter() - t0
        fpasses += 1
        examples_forwarded += int(xb.shape[0])

        v081.sync(device)
        t0 = time.perf_counter()
        loss.backward()
        v081.sync(device)
        bsec += time.perf_counter() - t0
        bpasses += 1
        examples_backwarded += int(xb.shape[0])

        v081.sync(device)
        t0 = time.perf_counter()
        opt.step()
        v081.sync(device)
        osec += time.perf_counter() - t0

        rows.append({
            'algorithm': 'adam',
            'update': step,
            'batch_cycle': step,
            'event': 'adam_update',
            'batch_loss_before_step': float(loss.detach().item()),
            'accepted': 1,
            'forward_passes': fpasses,
            'backward_passes': bpasses,
            'forward_seconds': fsec,
            'backward_seconds': bsec,
            'nn_compute_seconds': fsec + bsec,
            'calibrated_nn_compute_seconds': fpasses * pure_f + bpasses * pure_b,
        })

    v081.sync(device)
    wall = time.perf_counter() - wall_t0
    tr_loss, tr_acc = v081.evaluate(model, xtr, ytr, loss_fn)
    ev_loss, ev_acc = v081.evaluate(model, xev, yev, loss_fn)
    calibrated = fpasses * pure_f + bpasses * pure_b
    return rows, {
        'algorithm': 'adam',
        'final_train_loss': tr_loss,
        'final_train_accuracy': tr_acc,
        'final_eval_loss': ev_loss,
        'final_eval_accuracy': ev_acc,
        'accepted_updates': updates,
        'batch_cycles': updates,
        'forward_passes': fpasses,
        'backward_passes': bpasses,
        'forward_only_passes': fpasses - bpasses,
        'examples_forwarded': examples_forwarded,
        'examples_backwarded': examples_backwarded,
        'forward_seconds': fsec,
        'backward_seconds': bsec,
        'nn_compute_seconds': fsec + bsec,
        'optimizer_seconds': osec,
        'wall_seconds': wall,
        'calibrated_nn_compute_seconds': calibrated,
        'updates_per_backward': updates / max(bpasses, 1),
        'implementation_revision': 'stage4a-adam-minibatch',
    }


def run_agrls_batchlocked(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    max_updates: int,
    batch_size: int,
    seed: int,
    lr: float,
    pure_f: float,
    pure_b: float,
    compute_budget_seconds: Optional[float],
    label: str,
) -> Tuple[List[dict], dict]:
    """AGRLS-v0.12 development bridge to stochastic training.

    A local AGRLS cycle is *batch locked*: gradient, Adam anchor evaluation, and
    optional one-dimensional continuation all use the same mini-batch. No loss
    comparison, curvature estimate, or surrogate state crosses a mini-batch
    boundary. A new mini-batch therefore always begins with a fresh backward.

    This is intentionally conservative. It tests whether one mini-batch gradient
    can safely support two parameter updates without contaminating the surrogate
    with cross-batch sampling noise.
    """
    model.load_state_dict(init_state)
    model.to(device)
    theta = v081.params_to_vector(model).detach().clone()
    m = torch.zeros_like(theta)
    vv = torch.zeros_like(theta)
    adam_t = 0
    stream = make_batch_stream(int(xtr.shape[0]), batch_size, seed)

    rows: List[dict] = []
    accepted_updates = 0
    batch_cycles = 0
    continuation_attempts = 0
    continuation_accepts = 0
    nonmonotone_anchors = 0
    fpasses = bpasses = 0
    fsec = bsec = 0.0
    examples_forwarded = examples_backwarded = 0
    budget_stop = False
    budget_stop_reason = ''

    r = (pure_f + pure_b) / max(pure_f, 1e-15)
    mvr_threshold = 1.0 / max(r, 1.0)

    def calibrated_cost() -> float:
        return fpasses * pure_f + bpasses * pure_b

    def can_afford(add_f: int, add_b: int) -> bool:
        if compute_budget_seconds is None:
            return True
        return calibrated_cost() + add_f * pure_f + add_b * pure_b <= compute_budget_seconds + 1e-15

    v081.sync(device)
    wall_t0 = time.perf_counter()

    while accepted_updates < max_updates:
        # A useful cycle requires current F+B plus the mandatory anchor-evaluation F.
        if not can_afford(2, 1):
            budget_stop = True
            budget_stop_reason = 'insufficient_budget_for_new_batch_cycle'
            break

        idx = next(stream)
        xb, yb = batch_from_indices(xtr, ytr, idx, device)
        batch_cycles += 1

        f0, g, fs, bs = timed_value_grad(model, theta, xb, yb, loss_fn, device)
        fpasses += 1; bpasses += 1
        fsec += fs; bsec += bs
        examples_forwarded += int(xb.shape[0]); examples_backwarded += int(xb.shape[0])

        adam_t += 1
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * (g * g)
        mhat = m / (1.0 - 0.9 ** adam_t)
        vhat = vv / (1.0 - 0.999 ** adam_t)
        d = -mhat / (torch.sqrt(vhat) + 1e-8)
        nd = torch.linalg.vector_norm(d)
        if float(nd) <= 1e-15:
            break
        v1 = d / nd
        z1 = max(lr * float(nd), 1e-12)
        b1 = float(torch.dot(v1, g).item())
        theta_anchor = theta + z1 * v1

        f1, fs = timed_forward_value(model, theta_anchor, xb, yb, loss_fn, device)
        fpasses += 1; fsec += fs; examples_forwarded += int(xb.shape[0])
        actual1 = f0 - f1
        theta = theta_anchor
        accepted_updates += 1
        anchor_event = 'accept_adam_anchor' if actual1 > 0 else 'accept_adam_anchor_nonmonotone'
        if actual1 <= 0:
            nonmonotone_anchors += 1

        rows.append({
            'algorithm': label,
            'update': accepted_updates,
            'batch_cycle': batch_cycles,
            'event': anchor_event,
            'accepted': 1,
            'batch_loss_before_step': f0,
            'batch_loss_after_step': f1,
            'actual_improvement': actual1,
            'forward_passes': fpasses,
            'backward_passes': bpasses,
            'forward_seconds': fsec,
            'backward_seconds': bsec,
            'nn_compute_seconds': fsec + bsec,
            'calibrated_nn_compute_seconds': calibrated_cost(),
            'continuation_value_ratio': float('nan'),
            'mvr_threshold': mvr_threshold,
        })

        if accepted_updates >= max_updates:
            continue

        # Never carry the surrogate across batches. Only a monotone Adam anchor
        # on this exact batch may justify one continuation on the same batch.
        if actual1 <= 0 or b1 >= -1e-15:
            continue

        obs = [(z1, f1 - f0)]
        h1 = v081.refit_h_1d(b1, obs, 0.0, z1, scale=max(2.0 * z1, 1e-12))
        local_g = b1 + h1 * z1
        delta = z1
        if abs(local_g) <= 1e-12:
            continue
        if h1 > 1e-12:
            p_unc = -local_g / h1
            p = float(np.clip(p_unc, -delta, delta))
        else:
            p = -math.copysign(delta, local_g)
        predicted = -(local_g * p + 0.5 * h1 * p * p)
        ratio = predicted / max(actual1, 1e-15) if predicted > 0 and np.isfinite(predicted) else float('nan')

        if (not np.isfinite(ratio)) or ratio < mvr_threshold or abs(p) <= 1e-12:
            rows.append({
                'algorithm': label,
                'update': accepted_updates,
                'batch_cycle': batch_cycles,
                'event': 'skip_continuation_mvr',
                'accepted': 0,
                'batch_loss_before_step': f1,
                'batch_loss_after_step': f1,
                'predicted_improvement': predicted,
                'continuation_value_ratio': ratio,
                'mvr_threshold': mvr_threshold,
                'forward_passes': fpasses,
                'backward_passes': bpasses,
                'forward_seconds': fsec,
                'backward_seconds': bsec,
                'nn_compute_seconds': fsec + bsec,
                'calibrated_nn_compute_seconds': calibrated_cost(),
            })
            continue

        if not can_afford(1, 0):
            budget_stop = True
            budget_stop_reason = 'insufficient_budget_for_continuation_forward'
            break

        continuation_attempts += 1
        z2 = z1 + p
        theta2 = (theta_anchor - z1 * v1) + z2 * v1
        f2, fs = timed_forward_value(model, theta2, xb, yb, loss_fn, device)
        fpasses += 1; fsec += fs; examples_forwarded += int(xb.shape[0])
        actual2 = f1 - f2
        rho = actual2 / max(predicted, 1e-15)
        agreement = min(rho, 1.0 / rho) if rho > 0 and np.isfinite(rho) else 0.0

        if actual2 > 0:
            theta = theta2
            accepted_updates += 1
            continuation_accepts += 1
            event = 'accept_batchlocked_continuation'
            accepted = 1
        else:
            theta = theta_anchor
            event = 'reject_batchlocked_continuation'
            accepted = 0

        rows.append({
            'algorithm': label,
            'update': accepted_updates,
            'batch_cycle': batch_cycles,
            'event': event,
            'accepted': accepted,
            'batch_loss_before_step': f1,
            'batch_loss_after_step': f2,
            'actual_improvement': actual2,
            'predicted_improvement': predicted,
            'rho': rho,
            'agreement': agreement,
            'continuation_value_ratio': ratio,
            'mvr_threshold': mvr_threshold,
            'forward_passes': fpasses,
            'backward_passes': bpasses,
            'forward_seconds': fsec,
            'backward_seconds': bsec,
            'nn_compute_seconds': fsec + bsec,
            'calibrated_nn_compute_seconds': calibrated_cost(),
        })

        if budget_stop:
            break

    v081.copy_vector_to_params(model, theta)
    v081.sync(device)
    wall = time.perf_counter() - wall_t0
    tr_loss, tr_acc = v081.evaluate(model, xtr, ytr, loss_fn)
    ev_loss, ev_acc = v081.evaluate(model, xev, yev, loss_fn)
    calibrated = calibrated_cost()
    return rows, {
        'algorithm': label,
        'final_train_loss': tr_loss,
        'final_train_accuracy': tr_acc,
        'final_eval_loss': ev_loss,
        'final_eval_accuracy': ev_acc,
        'accepted_updates': accepted_updates,
        'batch_cycles': batch_cycles,
        'forward_passes': fpasses,
        'backward_passes': bpasses,
        'forward_only_passes': fpasses - bpasses,
        'examples_forwarded': examples_forwarded,
        'examples_backwarded': examples_backwarded,
        'continuation_attempts': continuation_attempts,
        'continuation_accepts': continuation_accepts,
        'continuation_accept_rate': continuation_accepts / max(continuation_attempts, 1),
        'nonmonotone_anchor_accepts': nonmonotone_anchors,
        'updates_per_backward': accepted_updates / max(bpasses, 1),
        'forward_seconds': fsec,
        'backward_seconds': bsec,
        'nn_compute_seconds': fsec + bsec,
        'wall_seconds': wall,
        'calibrated_nn_compute_seconds': calibrated,
        'compute_budget_seconds': compute_budget_seconds,
        'calibrated_budget_use_pct': (100.0 * calibrated / compute_budget_seconds) if compute_budget_seconds else float('nan'),
        'budget_stop': int(budget_stop),
        'budget_stop_reason': budget_stop_reason,
        'mvr_threshold': mvr_threshold,
        'surrogate_dimension': 1,
        'max_updates_per_batch_cycle': 2,
        'implementation_revision': 'v0.12-batchlocked-stochastic-bridge',
    }


def write_summary_plots(summaries: List[dict], out: Path) -> None:
    labels = [s['algorithm'] for s in summaries]
    acc = [100.0 * s['final_eval_accuracy'] for s in summaries]
    bwd = [s['backward_passes'] for s in summaries]
    comp = [s['calibrated_nn_compute_seconds'] for s in summaries]

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, x, y in zip(labels, bwd, acc):
        ax.scatter(x, y, s=80, label=name)
    ax.set_xlabel('Backward passes')
    ax.set_ylabel('Final evaluation accuracy (%)')
    ax.set_title('Stage 4A mini-batch diagnostic: accuracy vs backward passes')
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / 'final_accuracy_vs_backward_passes.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, x, y in zip(labels, comp, acc):
        ax.scatter(x, y, s=80, label=name)
    ax.set_xlabel('Calibrated pure mini-batch NN compute (s)')
    ax.set_ylabel('Final evaluation accuracy (%)')
    ax.set_title('Stage 4A mini-batch diagnostic: accuracy vs calibrated compute')
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / 'final_accuracy_vs_calibrated_compute.png', dpi=160); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['fashionmnist', 'mnist', 'fake'], default='fashionmnist')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--train-subset', type=int, default=10000)
    p.add_argument('--eval-subset', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--adam-updates', type=int, default=400)
    p.add_argument('--agrls-max-updates', type=int, default=700)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--adam-lr', type=float, default=0.001)
    p.add_argument('--out', default='stage4a_fashionmnist_cnn_minibatch_locked_seed0')
    args = p.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print('CUDA requested but unavailable; using CPU.')
    else:
        device = torch.device(args.device)

    v081.set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xev, yev = v081.load_fullbatch_data(
        args.dataset, Path(args.data_dir), args.train_subset, args.eval_subset, args.seed, device
    )
    loss_fn = nn.CrossEntropyLoss()
    model = SmallFashionCNN().to(device)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n_params = parameter_count(model)

    # Shared first batch for reporting-only pure mini-batch calibration.
    calib_stream = make_batch_stream(int(xtr.shape[0]), args.batch_size, args.seed)
    cidx = next(calib_stream)
    cxb, cyb = batch_from_indices(xtr, ytr, cidx, device)
    pure = calibrate_pure_batch_costs(model, init_state, cxb, cyb, loss_fn, device)
    pure_f = float(pure['pure_forward_median_s'])
    pure_b = float(pure['pure_backward_median_s'])
    adam_budget = args.adam_updates * (pure_f + pure_b)

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}')
    print(f'Model parameters: {n_params:,}')
    print(f'Batch size: {args.batch_size}')
    print(f'Pure mini-batch forward: {pure_f:.6f}s')
    print(f'Pure mini-batch backward: {pure_b:.6f}s')
    print(f'Full-gradient / forward ratio: {pure["pure_full_gradient_forward_ratio"]:.3f}')
    print(f'Adam calibrated compute budget ({args.adam_updates} updates): {adam_budget:.6f}s')

    summaries: List[dict] = []
    traces: Dict[str, List[dict]] = {}

    print('Running adam ...')
    rows, s = run_adam_minibatch(
        model, init_state, xtr, ytr, xev, yev, loss_fn, device,
        args.adam_updates, args.batch_size, args.seed, args.adam_lr, pure_f, pure_b
    )
    summaries.append(s); traces['adam'] = rows; v081.write_csv(out / 'trace_adam.csv', rows)

    print('Running AGRLS batch-locked fixed-update ...')
    rows, s = run_agrls_batchlocked(
        model, init_state, xtr, ytr, xev, yev, loss_fn, device,
        args.adam_updates, args.batch_size, args.seed, args.adam_lr, pure_f, pure_b,
        compute_budget_seconds=None, label='agrls_batchlocked_fixed'
    )
    summaries.append(s); traces['agrls_batchlocked_fixed'] = rows
    v081.write_csv(out / 'trace_agrls_batchlocked_fixed.csv', rows)

    print('Running AGRLS batch-locked equal-compute ...')
    rows, s = run_agrls_batchlocked(
        model, init_state, xtr, ytr, xev, yev, loss_fn, device,
        args.agrls_max_updates, args.batch_size, args.seed, args.adam_lr, pure_f, pure_b,
        compute_budget_seconds=adam_budget, label='agrls_batchlocked_equal_compute'
    )
    summaries.append(s); traces['agrls_batchlocked_equal_compute'] = rows
    v081.write_csv(out / 'trace_agrls_batchlocked_equal_compute.csv', rows)

    batches_per_epoch = int(xtr.shape[0]) // args.batch_size
    for s in summaries:
        s.update({
            'seed': args.seed,
            'dataset': args.dataset,
            'model': 'SmallFashionCNN_32_64_128',
            'model_parameters': n_params,
            'train_subset': int(xtr.shape[0]),
            'eval_subset': int(xev.shape[0]),
            'batch_size': args.batch_size,
            'drop_last': 1,
            'batches_per_epoch': batches_per_epoch,
            'batch_cycles_epoch_equivalent': s['batch_cycles'] / max(batches_per_epoch, 1),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
            'adam_lr': args.adam_lr,
            'calib_pure_forward_median_s': pure_f,
            'calib_pure_backward_median_s': pure_b,
            'calib_pure_full_gradient_forward_ratio': pure['pure_full_gradient_forward_ratio'],
            'adam_reference_budget_seconds': adam_budget,
        })
    v081.write_csv(out / 'summary.csv', summaries)

    by = {s['algorithm']: s for s in summaries}
    paired: List[dict] = []
    a = by['adam']
    for name in ['agrls_batchlocked_fixed', 'agrls_batchlocked_equal_compute']:
        g = by[name]
        paired.append({
            'algorithm': name,
            'accuracy_delta_pp': 100.0 * (g['final_eval_accuracy'] - a['final_eval_accuracy']),
            'backward_reduction_pct': 100.0 * (1.0 - g['backward_passes'] / a['backward_passes']),
            'calibrated_nn_reduction_pct': 100.0 * (1.0 - g['calibrated_nn_compute_seconds'] / a['calibrated_nn_compute_seconds']),
            'direct_nn_time_reduction_pct': 100.0 * (1.0 - g['nn_compute_seconds'] / a['nn_compute_seconds']),
            'wall_time_reduction_pct': 100.0 * (1.0 - g['wall_seconds'] / a['wall_seconds']),
            'extra_updates': g['accepted_updates'] - a['accepted_updates'],
            'batch_cycle_delta': g['batch_cycles'] - a['batch_cycles'],
            'updates_per_backward': g['updates_per_backward'],
            'calibrated_budget_use_pct': 100.0 * g['calibrated_nn_compute_seconds'] / a['calibrated_nn_compute_seconds'],
        })
    v081.write_csv(out / 'paired_vs_adam.csv', paired)
    write_summary_plots(summaries, out)

    report: List[str] = []
    report.append('AGRLS Stage 4A v0.12 — batch-locked stochastic bridge development\n')
    report.append(f'dataset={args.dataset}; train_n={xtr.shape[0]}; eval_n={xev.shape[0]}; seed={args.seed}; batch_size={args.batch_size}\n')
    report.append(f'model=SmallFashionCNN_32_64_128; parameters={n_params}; adam_lr={args.adam_lr}\n')
    report.append(f'device={device}; gpu={torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"}\n')
    report.append(f'pure_batch_forward={pure_f:.6f}s; pure_batch_backward={pure_b:.6f}s; full_gradient/forward={pure["pure_full_gradient_forward_ratio"]:.6f}\n')
    report.append('IMPORTANT: AGRLS local loss/curvature comparisons are batch-locked. No surrogate state crosses a mini-batch boundary.\n\n')
    for s in summaries:
        report.append(
            f"{s['algorithm']:36s} eval_acc={s['final_eval_accuracy']:.4f} train_loss={s['final_train_loss']:.5f} "
            f"updates={s['accepted_updates']} cycles={s['batch_cycles']} fwd={s['forward_passes']} bwd={s['backward_passes']} "
            f"calib_nn_s={s['calibrated_nn_compute_seconds']:.5f} direct_nn_s={s['nn_compute_seconds']:.5f} wall_s={s['wall_seconds']:.5f}\n"
        )
    report.append('\nPaired versus Adam:\n')
    for q in paired:
        report.append(
            f"{q['algorithm']}: accuracy_delta_pp={q['accuracy_delta_pp']:.4f}; "
            f"backward_reduction_pct={q['backward_reduction_pct']:.2f}; "
            f"calibrated_nn_reduction_pct={q['calibrated_nn_reduction_pct']:.2f}; "
            f"extra_updates={q['extra_updates']}; budget_use_pct={q['calibrated_budget_use_pct']:.2f}\n"
        )
    (out / 'REPORT.txt').write_text(''.join(report), encoding='utf-8')
    print('\n' + ''.join(report))
    print('Results:', out.resolve())


if __name__ == '__main__':
    main()
