from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

import agrls_stage1_mnist_fullbatch_v081 as v081
import agrls_stage4a_minibatch_locked_v012 as v012


# Stage 4G: forward-gated gradient refresh (FGGR)
# ------------------------------------------------
# This is a development pivot after Stage 4F showed that the v0.14 one-dimensional
# quadratic continuation did not outperform simpler same-direction reuse on the
# seen Fashion-MNIST seeds. The goal here is to return to the project's more
# distinctive question: can a forward-only test on the NEW mini-batch decide
# whether the next backward pass is worth paying for?
#
# Every batch produces exactly ONE parameter update.
# - Adam:               F+B -> fresh Adam update.
# - periodic_k2:        fresh F+B, then on the next batch F only + reuse last
#                       Adam displacement, alternating deterministically.
# - fggr_monotone:      after a fresh gradient, the next batch is eligible for
#                       a forward-first stale-direction trial. Compute f(theta)
#                       with graph retained, evaluate f(theta + d_stale) by a
#                       stateless forward, and:
#                         * accept stale update and SKIP backward if loss falls;
#                         * otherwise backward through the already-paid f(theta)
#                           graph and take a fresh Adam update.
#                       After an accepted skip, force a fresh gradient on the
#                       following batch (max one consecutive skipped backward).
#
# Crucially, the gate compares two losses on the SAME new batch. No loss value or
# surrogate state is compared across batch boundaries.


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def named_trainable(model: nn.Module):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def init_adam_state(model: nn.Module):
    state = {}
    for n, p in named_trainable(model):
        state[n] = {
            'm': torch.zeros_like(p),
            'v': torch.zeros_like(p),
        }
    return state


@torch.no_grad()
def adam_displacement_from_current_grads(
    model: nn.Module,
    state: Dict[str, Dict[str, torch.Tensor]],
    adam_t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    bc1 = 1.0 - beta1 ** adam_t
    bc2 = 1.0 - beta2 ** adam_t
    for n, p in named_trainable(model):
        if p.grad is None:
            raise RuntimeError(f'Missing gradient for parameter {n}')
        g = p.grad.detach()
        s = state[n]
        s['m'].mul_(beta1).add_(g, alpha=1.0 - beta1)
        s['v'].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        mhat = s['m'] / bc1
        vhat = s['v'] / bc2
        out[n] = (-lr * mhat / (torch.sqrt(vhat) + eps)).detach().clone()
    return out


@torch.no_grad()
def apply_displacement(model: nn.Module, displacement: Dict[str, torch.Tensor]) -> None:
    for n, p in named_trainable(model):
        p.add_(displacement[n])


def displacement_norm(displacement: Dict[str, torch.Tensor]) -> float:
    total = None
    for d in displacement.values():
        x = torch.sum(d.detach().double() * d.detach().double())
        total = x if total is None else total + x
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


def timed_forward_graph(model, xb, yb, loss_fn, device):
    model.zero_grad(set_to_none=True)
    v081.sync(device)
    t0 = time.perf_counter()
    loss = loss_fn(model(xb), yb)
    v081.sync(device)
    sec = time.perf_counter() - t0
    return loss, float(loss.detach().item()), sec


def timed_forward_nograd(model, xb, yb, loss_fn, device):
    v081.sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        val = float(loss_fn(model(xb), yb).item())
    v081.sync(device)
    return val, time.perf_counter() - t0


def timed_trial_functional(
    model: nn.Module,
    displacement: Dict[str, torch.Tensor],
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
):
    # Keep the live model parameters unchanged so an already-built f(theta)
    # autograd graph can still be upgraded to backward if the stale trial fails.
    candidate_params = {
        n: (p.detach() + displacement[n]).detach()
        for n, p in named_trainable(model)
    }
    buffers = {n: b for n, b in model.named_buffers()}
    v081.sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = functional_call(model, (candidate_params, buffers), (xb,), strict=False)
        val = float(loss_fn(logits, yb).item())
    v081.sync(device)
    sec = time.perf_counter() - t0
    return val, sec


def timed_backward(loss: torch.Tensor, device: torch.device) -> float:
    v081.sync(device)
    t0 = time.perf_counter()
    loss.backward()
    v081.sync(device)
    return time.perf_counter() - t0


def evaluate_summary(
    model: nn.Module,
    xtr, ytr, xev, yev,
    loss_fn,
    label: str,
    batch_cycles: int,
    updates: int,
    fpasses: int,
    bpasses: int,
    fsec: float,
    bsec: float,
    wall: float,
    pure_f: float,
    pure_b: float,
    examples_forwarded: int,
    examples_backwarded: int,
    reuse_attempts: int,
    reuse_accepts: int,
    reuse_rejects: int,
    forced_refreshes: int,
    implementation_revision: str,
) -> dict:
    tr_loss, tr_acc = v081.evaluate(model, xtr, ytr, loss_fn)
    ev_loss, ev_acc = v081.evaluate(model, xev, yev, loss_fn)
    calibrated = fpasses * pure_f + bpasses * pure_b
    return {
        'algorithm': label,
        'final_train_loss': tr_loss,
        'final_train_accuracy': tr_acc,
        'final_eval_loss': ev_loss,
        'final_eval_accuracy': ev_acc,
        'accepted_updates': updates,
        'batch_cycles': batch_cycles,
        'forward_passes': fpasses,
        'backward_passes': bpasses,
        'forward_only_passes': fpasses - bpasses,
        'updates_per_backward': updates / max(bpasses, 1),
        'forward_seconds': fsec,
        'backward_seconds': bsec,
        'nn_compute_seconds': fsec + bsec,
        'wall_seconds': wall,
        'calibrated_nn_compute_seconds': calibrated,
        'examples_forwarded': examples_forwarded,
        'examples_backwarded': examples_backwarded,
        'reuse_attempts': reuse_attempts,
        'reuse_accepts': reuse_accepts,
        'reuse_rejects': reuse_rejects,
        'reuse_accept_rate': reuse_accepts / max(reuse_attempts, 1),
        'forced_refreshes': forced_refreshes,
        'implementation_revision': implementation_revision,
    }


def run_adam_tensorwise(
    model, init_state, xtr, ytr, xev, yev, loss_fn, device,
    updates, batch_size, seed, lr, pure_f, pure_b,
):
    model.load_state_dict(init_state)
    model.to(device)
    model.train()
    state = init_adam_state(model)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)

    rows: List[dict] = []
    fpasses = bpasses = 0
    fsec = bsec = 0.0
    exf = exb = 0
    adam_t = 0

    v081.sync(device)
    wall0 = time.perf_counter()
    for u in range(1, updates + 1):
        idx = next(stream)
        xb, yb = v012.batch_from_indices(xtr, ytr, idx, device)
        loss, f0, fs = timed_forward_graph(model, xb, yb, loss_fn, device)
        bs = timed_backward(loss, device)
        fpasses += 1; bpasses += 1; fsec += fs; bsec += bs
        exf += int(xb.shape[0]); exb += int(xb.shape[0])
        adam_t += 1
        disp = adam_displacement_from_current_grads(model, state, adam_t, lr)
        apply_displacement(model, disp)
        rows.append({
            'algorithm': 'adam_tensorwise', 'update': u, 'batch_cycle': u,
            'event': 'fresh_adam', 'batch_loss': f0,
            'forward_passes': fpasses, 'backward_passes': bpasses,
            'forward_seconds': fsec, 'backward_seconds': bsec,
            'calibrated_nn_compute_seconds': fpasses * pure_f + bpasses * pure_b,
            'displacement_norm': displacement_norm(disp),
        })
    v081.sync(device)
    wall = time.perf_counter() - wall0
    summary = evaluate_summary(
        model, xtr, ytr, xev, yev, loss_fn, 'adam_tensorwise', updates, updates,
        fpasses, bpasses, fsec, bsec, wall, pure_f, pure_b, exf, exb,
        0, 0, 0, updates, 'stage4g-tensorwise-adam',
    )
    return rows, summary


def run_periodic_k2(
    model, init_state, xtr, ytr, xev, yev, loss_fn, device,
    updates, batch_size, seed, lr, pure_f, pure_b,
):
    model.load_state_dict(init_state)
    model.to(device)
    model.train()
    state = init_adam_state(model)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)

    rows: List[dict] = []
    fpasses = bpasses = 0
    fsec = bsec = 0.0
    exf = exb = 0
    adam_t = 0
    last_disp: Optional[Dict[str, torch.Tensor]] = None
    reuse_next = False
    reuse_accepts = 0
    forced_refreshes = 0

    v081.sync(device)
    wall0 = time.perf_counter()
    for u in range(1, updates + 1):
        idx = next(stream)
        xb, yb = v012.batch_from_indices(xtr, ytr, idx, device)
        if reuse_next and last_disp is not None:
            f0, fs = timed_forward_nograd(model, xb, yb, loss_fn, device)
            fpasses += 1; fsec += fs; exf += int(xb.shape[0])
            apply_displacement(model, last_disp)
            reuse_accepts += 1
            reuse_next = False
            event = 'blind_periodic_reuse'
            dn = displacement_norm(last_disp)
        else:
            loss, f0, fs = timed_forward_graph(model, xb, yb, loss_fn, device)
            bs = timed_backward(loss, device)
            fpasses += 1; bpasses += 1; fsec += fs; bsec += bs
            exf += int(xb.shape[0]); exb += int(xb.shape[0])
            adam_t += 1
            last_disp = adam_displacement_from_current_grads(model, state, adam_t, lr)
            apply_displacement(model, last_disp)
            reuse_next = True
            forced_refreshes += 1
            event = 'fresh_adam'
            dn = displacement_norm(last_disp)
        rows.append({
            'algorithm': 'periodic_k2_reuse', 'update': u, 'batch_cycle': u,
            'event': event, 'batch_loss': f0,
            'forward_passes': fpasses, 'backward_passes': bpasses,
            'forward_seconds': fsec, 'backward_seconds': bsec,
            'calibrated_nn_compute_seconds': fpasses * pure_f + bpasses * pure_b,
            'displacement_norm': dn,
        })
    v081.sync(device)
    wall = time.perf_counter() - wall0
    attempts = updates // 2
    summary = evaluate_summary(
        model, xtr, ytr, xev, yev, loss_fn, 'periodic_k2_reuse', updates, updates,
        fpasses, bpasses, fsec, bsec, wall, pure_f, pure_b, exf, exb,
        attempts, reuse_accepts, 0, forced_refreshes,
        'stage4g-tensorwise-periodic-k2-stale-displacement',
    )
    return rows, summary


def run_fggr_monotone(
    model, init_state, xtr, ytr, xev, yev, loss_fn, device,
    updates, batch_size, seed, lr, pure_f, pure_b,
):
    model.load_state_dict(init_state)
    model.to(device)
    model.train()
    state = init_adam_state(model)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)

    rows: List[dict] = []
    fpasses = bpasses = 0
    fsec = bsec = 0.0
    exf = exb = 0
    adam_t = 0
    last_disp: Optional[Dict[str, torch.Tensor]] = None
    eligible_reuse = False
    reuse_attempts = reuse_accepts = reuse_rejects = 0
    forced_refreshes = 0

    v081.sync(device)
    wall0 = time.perf_counter()
    for u in range(1, updates + 1):
        idx = next(stream)
        xb, yb = v012.batch_from_indices(xtr, ytr, idx, device)

        if eligible_reuse and last_disp is not None:
            # First forward is at the current point and keeps its graph. If the
            # stale-direction trial fails, we upgrade THIS graph to backward.
            loss0, f0, fs0 = timed_forward_graph(model, xb, yb, loss_fn, device)
            fpasses += 1; fsec += fs0; exf += int(xb.shape[0])

            f1, fs1 = timed_trial_functional(model, last_disp, xb, yb, loss_fn, device)
            fpasses += 1; fsec += fs1; exf += int(xb.shape[0])
            reuse_attempts += 1

            if np.isfinite(f1) and f1 < f0:
                apply_displacement(model, last_disp)
                reuse_accepts += 1
                eligible_reuse = False  # cap: never skip two backwards in a row
                event = 'accept_stale_forward_gated_skip_backward'
                used_loss = f1
                dn = displacement_norm(last_disp)
            else:
                bs = timed_backward(loss0, device)
                bpasses += 1; bsec += bs; exb += int(xb.shape[0])
                adam_t += 1
                last_disp = adam_displacement_from_current_grads(model, state, adam_t, lr)
                apply_displacement(model, last_disp)
                reuse_rejects += 1
                eligible_reuse = True
                event = 'reject_stale_trial_upgrade_cached_forward_to_backward'
                used_loss = f0
                dn = displacement_norm(last_disp)
        else:
            # Forced refresh after a successful skip, or initialization.
            loss0, f0, fs0 = timed_forward_graph(model, xb, yb, loss_fn, device)
            bs = timed_backward(loss0, device)
            fpasses += 1; bpasses += 1; fsec += fs0; bsec += bs
            exf += int(xb.shape[0]); exb += int(xb.shape[0])
            adam_t += 1
            last_disp = adam_displacement_from_current_grads(model, state, adam_t, lr)
            apply_displacement(model, last_disp)
            eligible_reuse = True
            forced_refreshes += 1
            event = 'forced_fresh_adam'
            used_loss = f0
            dn = displacement_norm(last_disp)

        rows.append({
            'algorithm': 'fggr_monotone', 'update': u, 'batch_cycle': u,
            'event': event, 'batch_loss': used_loss,
            'forward_passes': fpasses, 'backward_passes': bpasses,
            'forward_seconds': fsec, 'backward_seconds': bsec,
            'calibrated_nn_compute_seconds': fpasses * pure_f + bpasses * pure_b,
            'reuse_attempts': reuse_attempts,
            'reuse_accepts': reuse_accepts,
            'reuse_rejects': reuse_rejects,
            'reuse_accept_rate': reuse_accepts / max(reuse_attempts, 1),
            'displacement_norm': dn,
        })

    v081.sync(device)
    wall = time.perf_counter() - wall0
    summary = evaluate_summary(
        model, xtr, ytr, xev, yev, loss_fn, 'fggr_monotone', updates, updates,
        fpasses, bpasses, fsec, bsec, wall, pure_f, pure_b, exf, exb,
        reuse_attempts, reuse_accepts, reuse_rejects, forced_refreshes,
        'stage4g-tensorwise-forward-gated-gradient-refresh-cap1',
    )
    return rows, summary


def run_one(alg, *args, **kwargs):
    if alg == 'adam_tensorwise':
        return run_adam_tensorwise(*args, **kwargs)
    if alg == 'periodic_k2_reuse':
        return run_periodic_k2(*args, **kwargs)
    if alg == 'fggr_monotone':
        return run_fggr_monotone(*args, **kwargs)
    raise ValueError(alg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--train-subset', type=int, default=10000)
    p.add_argument('--eval-subset', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--updates', type=int, default=400)
    p.add_argument('--seed', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--adam-lr', type=float, default=0.001)
    p.add_argument('--out', default='stage4g_forward_gated_seed64')
    p.add_argument('--algorithms', nargs='+', default=['adam_tensorwise','periodic_k2_reuse','fggr_monotone'])
    args = p.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else args.device)
    if args.device == 'cuda' and device.type != 'cuda':
        print('CUDA requested but unavailable; using CPU.')

    v081.set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xev, yev = v081.load_fullbatch_data(
        args.dataset, Path(args.data_dir), args.train_subset, args.eval_subset, args.seed, device
    )
    loss_fn = nn.CrossEntropyLoss()
    model = v012.SmallFashionCNN().to(device)
    init_state = clone_state_dict(model)
    n_params = v012.parameter_count(model)

    calib_stream = v012.make_batch_stream(int(xtr.shape[0]), args.batch_size, args.seed)
    cidx = next(calib_stream)
    cxb, cyb = v012.batch_from_indices(xtr, ytr, cidx, device)
    pure = v012.calibrate_pure_batch_costs(model, init_state, cxb, cyb, loss_fn, device)
    pure_f = float(pure['pure_forward_median_s'])
    pure_b = float(pure['pure_backward_median_s'])

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}')
    print(f'Seed={args.seed} | params={n_params:,} | batch={args.batch_size} | updates={args.updates}')
    print(f'Pure F={pure_f:.6f}s | pure B={pure_b:.6f}s | (F+B)/F={pure["pure_full_gradient_forward_ratio"]:.3f}')
    print('Execution order:', args.algorithms)

    summaries = []
    for alg in args.algorithms:
        print('RUNNING', alg)
        rows, s = run_one(
            alg, model, init_state, xtr, ytr, xev, yev, loss_fn, device,
            args.updates, args.batch_size, args.seed, args.adam_lr, pure_f, pure_b
        )
        s.update({
            'seed': args.seed,
            'dataset': args.dataset,
            'model': 'SmallFashionCNN_32_64_128',
            'model_parameters': n_params,
            'train_subset': int(xtr.shape[0]),
            'eval_subset': int(xev.shape[0]),
            'batch_size': args.batch_size,
            'adam_lr': args.adam_lr,
            'calib_pure_forward_median_s': pure_f,
            'calib_pure_backward_median_s': pure_b,
            'calib_pure_full_gradient_forward_ratio': pure['pure_full_gradient_forward_ratio'],
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
            'execution_order': ' > '.join(args.algorithms),
        })
        summaries.append(s)
        v081.write_csv(out / f'trace_{alg}.csv', rows)

    v081.write_csv(out / 'summary.csv', summaries)
    by = {s['algorithm']: s for s in summaries}
    a = by['adam_tensorwise']
    paired = []
    for s in summaries:
        if s['algorithm'] == 'adam_tensorwise':
            continue
        paired.append({
            'seed': args.seed,
            'algorithm': s['algorithm'],
            'execution_order': s['execution_order'],
            'adam_eval_accuracy': a['final_eval_accuracy'],
            'method_eval_accuracy': s['final_eval_accuracy'],
            'accuracy_delta_pp': 100.0 * (s['final_eval_accuracy'] - a['final_eval_accuracy']),
            'adam_backward_passes': a['backward_passes'],
            'method_backward_passes': s['backward_passes'],
            'backward_reduction_pct': 100.0 * (a['backward_passes'] - s['backward_passes']) / max(a['backward_passes'],1),
            'adam_calibrated_nn_seconds': a['calibrated_nn_compute_seconds'],
            'method_calibrated_nn_seconds': s['calibrated_nn_compute_seconds'],
            'calibrated_nn_reduction_pct': 100.0 * (a['calibrated_nn_compute_seconds'] - s['calibrated_nn_compute_seconds']) / max(a['calibrated_nn_compute_seconds'],1e-15),
            'adam_direct_nn_seconds': a['nn_compute_seconds'],
            'method_direct_nn_seconds': s['nn_compute_seconds'],
            'direct_nn_time_reduction_pct': 100.0 * (a['nn_compute_seconds'] - s['nn_compute_seconds']) / max(a['nn_compute_seconds'],1e-15),
            'adam_wall_seconds': a['wall_seconds'],
            'method_wall_seconds': s['wall_seconds'],
            'wall_time_reduction_pct': 100.0 * (a['wall_seconds'] - s['wall_seconds']) / max(a['wall_seconds'],1e-15),
            'reuse_attempts': s['reuse_attempts'],
            'reuse_accepts': s['reuse_accepts'],
            'reuse_rejects': s['reuse_rejects'],
            'reuse_accept_rate': s['reuse_accept_rate'],
        })
    v081.write_csv(out / 'paired_vs_adam.csv', paired)

    report = []
    report.append('AGRLS Stage 4G - forward-gated gradient refresh development\n')
    report.append(f'dataset={args.dataset}; seed={args.seed}; train={len(xtr)}; eval={len(xev)}; batch={args.batch_size}; updates={args.updates}\n')
    report.append('All methods process exactly one new mini-batch and make exactly one parameter update per batch cycle.\n')
    report.append('FGGR gate: stale Adam displacement is tested by a same-batch forward; backward is skipped only if candidate loss is lower. After a successful skip, the next batch forces a fresh gradient.\n\n')
    for s in summaries:
        report.append(
            f"{s['algorithm']:20s} eval_acc={s['final_eval_accuracy']:.4f} "
            f"bwd={s['backward_passes']} fwd={s['forward_passes']} "
            f"calib={s['calibrated_nn_compute_seconds']:.6f}s wall={s['wall_seconds']:.6f}s "
            f"reuse_accept={s['reuse_accepts']}/{s['reuse_attempts']}\n"
        )
    (out / 'REPORT.txt').write_text(''.join(report), encoding='utf-8')
    print(''.join(report))


if __name__ == '__main__':
    main()
