#!/usr/bin/env python3
"""
AGRLS Stage 1D — information-matched 1-D probe-free neural-network diagnostic
=================================================

Purpose
-------
Move from synthetic objectives to a real nonlinear neural network while
preserving a deterministic objective. The first gate uses a full-batch subset
of MNIST so that surrogate error reflects network geometry rather than
mini-batch sampling noise.

This is a diagnostic, not a publication benchmark. Standard optimizer
hyperparameters are deliberately simple and must later be tuned fairly.

Key comparison
--------------
- SGD
- SGD + momentum
- Adam
- AGRLS fixed-1 (internal control: refresh after every accepted move)
- AGRLS MVR (marginal-value refresh, max 3 accepted moves/gradient)

Unlike Stage 0, the MVR cost ratio is measured on the user's actual hardware:
    r = median(value+gradient query time) / median(value-only query time)

The legacy MVR controller uses threshold = 1 / (r + 4), because each refresh
requires one gradient query, three map probes, and a trial forward.

The new probe-free controller (AGRLS-PF-MVR) transfers curvature from the
previous subspace and updates it only from forward evaluations that would
already be required to test candidate moves. Its refresh threshold is
    threshold = 1 / (r + 1)
because a refresh now costs one gradient query plus the next trial forward.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def params_to_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def grads_to_vector(model: nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def copy_vector_to_params(model: nn.Module, vec: torch.Tensor) -> None:
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[offset:offset+n].view_as(p))
            offset += n
    if offset != vec.numel():
        raise RuntimeError("Parameter vector length mismatch")


def orthonormal_basis(
    g: torch.Tensor,
    prev_step: Optional[torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    eps = 1e-15
    ng = torch.linalg.vector_norm(g)
    if float(ng) <= eps:
        v1 = torch.randn(g.numel(), device=g.device, dtype=g.dtype, generator=generator)
        v1 = v1 / torch.linalg.vector_norm(v1)
    else:
        v1 = -g / ng

    if prev_step is not None and float(torch.linalg.vector_norm(prev_step)) > eps:
        v2 = prev_step - torch.dot(prev_step, v1) * v1
        n2 = torch.linalg.vector_norm(v2)
    else:
        n2 = torch.tensor(0.0, device=g.device)
        v2 = torch.empty_like(g)

    if float(n2) <= 1e-10:
        for _ in range(20):
            cand = torch.randn(g.numel(), device=g.device, dtype=g.dtype, generator=generator)
            cand = cand - torch.dot(cand, v1) * v1
            nc = torch.linalg.vector_norm(cand)
            if float(nc) > 1e-10:
                v2 = cand / nc
                break
        else:
            raise RuntimeError("Could not construct second subspace direction")
    else:
        v2 = v2 / n2

    return torch.stack([v1, v2], dim=1)  # [d,2]


# ---------------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_fullbatch_data(
    dataset_name: str,
    root: Path,
    train_subset: int,
    eval_subset: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed + 7001)

    if dataset_name in {"mnist", "fashionmnist"}:
        ds_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST

        # torchvision's FashionMNIST class historically points to a single
        # HTTP S3 website mirror. That endpoint can time out even when the
        # dataset itself is available. For Fashion-MNIST only, prepend the
        # upstream GitHub raw-data mirror and retain torchvision's original
        # mirror as a fallback. This changes data acquisition only; dataset
        # bytes, sampling, model, optimizer, seeds, and experimental logic
        # are unchanged.
        if dataset_name == "fashionmnist":
            github_mirror = (
                "https://raw.githubusercontent.com/"
                "zalandoresearch/fashion-mnist/master/data/fashion/"
            )
            original_mirrors = list(getattr(datasets.FashionMNIST, "mirrors", []))
            datasets.FashionMNIST.mirrors = [github_mirror] + [
                m for m in original_mirrors if m != github_mirror
            ]

        tr = ds_cls(root=str(root), train=True, download=True)
        te = ds_cls(root=str(root), train=False, download=True)
        train_subset = min(train_subset, len(tr))
        eval_subset = min(eval_subset, len(te))
        idx_tr = torch.randperm(len(tr), generator=gen)[:train_subset]
        idx_te = torch.arange(eval_subset)
        xtr = tr.data[idx_tr].float().unsqueeze(1).div_(255.0)
        ytr = tr.targets[idx_tr].long()
        xev = te.data[idx_te].float().unsqueeze(1).div_(255.0)
        yev = te.targets[idx_te].long()
    elif dataset_name == "fake":
        # Development-only smoke path; no download required.
        train_subset = min(train_subset, 2048)
        eval_subset = min(eval_subset, 512)
        xtr = torch.randn(train_subset, 1, 28, 28, generator=gen)
        ytr = torch.randint(0, 10, (train_subset,), generator=gen)
        xev = torch.randn(eval_subset, 1, 28, 28, generator=gen)
        yev = torch.randint(0, 10, (eval_subset,), generator=gen)
    else:
        raise ValueError(dataset_name)

    return (
        xtr.to(device, non_blocking=False),
        ytr.to(device, non_blocking=False),
        xev.to(device, non_blocking=False),
        yev.to(device, non_blocking=False),
    )


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> Tuple[float, float]:
    model.eval()
    logits = model(x)
    loss = float(loss_fn(logits, y).item())
    acc = float((logits.argmax(dim=1) == y).float().mean().item())
    return loss, acc


# ---------------------------------------------------------------------------
# Timed objective oracle
# ---------------------------------------------------------------------------

class NeuralOracle:
    def __init__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, device: torch.device):
        self.model = model
        self.x = x
        self.y = y
        self.loss_fn = loss_fn
        self.device = device
        self.forward_calls = 0
        self.gradient_calls = 0
        self.forward_seconds = 0.0
        self.gradient_seconds = 0.0

    def value(self, theta: torch.Tensor, count: bool = True) -> float:
        sync(self.device)
        t0 = time.perf_counter()
        copy_vector_to_params(self.model, theta)
        self.model.eval()
        with torch.no_grad():
            loss = self.loss_fn(self.model(self.x), self.y)
        sync(self.device)
        dt = time.perf_counter() - t0
        if count:
            self.forward_calls += 1
            self.forward_seconds += dt
        return float(loss.item())

    def value_grad(self, theta: torch.Tensor, count: bool = True) -> Tuple[float, torch.Tensor]:
        sync(self.device)
        t0 = time.perf_counter()
        copy_vector_to_params(self.model, theta)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(self.x), self.y)
        loss.backward()
        g = grads_to_vector(self.model).detach().clone()
        sync(self.device)
        dt = time.perf_counter() - t0
        if count:
            self.gradient_calls += 1
            self.gradient_seconds += dt
        return float(loss.item()), g


def calibrate_query_costs(oracle: NeuralOracle, theta: torch.Tensor, warmup: int = 3, reps: int = 7) -> dict:
    for _ in range(warmup):
        oracle.value(theta, count=False)
        oracle.value_grad(theta, count=False)
    f_times = []
    g_times = []
    for _ in range(reps):
        sync(oracle.device)
        t0 = time.perf_counter()
        oracle.value(theta, count=False)
        sync(oracle.device)
        f_times.append(time.perf_counter() - t0)

        sync(oracle.device)
        t0 = time.perf_counter()
        oracle.value_grad(theta, count=False)
        sync(oracle.device)
        g_times.append(time.perf_counter() - t0)

    fmed = statistics.median(f_times)
    gmed = statistics.median(g_times)
    return {
        "forward_query_median_s": fmed,
        "gradient_query_median_s": gmed,
        "measured_gradient_cost_ratio": gmed / max(fmed, 1e-12),
        "forward_query_times_s": f_times,
        "gradient_query_times_s": g_times,
    }


# ---------------------------------------------------------------------------
# Small 2-D surrogate mathematics
# ---------------------------------------------------------------------------

def solve_full_trust_region(g: np.ndarray, H: np.ndarray, delta: float) -> np.ndarray:
    k = len(g)
    if delta <= 0:
        return np.zeros(k)
    H = 0.5 * (H + H.T)
    lam, Q = np.linalg.eigh(H)
    gh = Q.T @ g
    eps = 1e-10
    if np.min(lam) > eps:
        p_unc = Q @ (-gh / lam)
        if np.linalg.norm(p_unc) <= delta:
            return p_unc
    lower = max(0.0, -float(np.min(lam)) + 1e-9)

    def ph(mu: float) -> np.ndarray:
        den = lam + mu
        den = np.where(np.abs(den) < eps, eps, den)
        return -gh / den

    p0h = ph(lower)
    n0 = np.linalg.norm(p0h)
    if np.isfinite(n0) and n0 < delta and np.min(lam) <= 0:
        idx = int(np.argmin(lam))
        rem = max(0.0, delta * delta - n0 * n0)
        sign = -1.0 if gh[idx] >= 0 else 1.0
        p0h[idx] += sign * math.sqrt(rem)
        return Q @ p0h

    high = max(1.0, lower + 1.0)
    while np.linalg.norm(ph(high)) > delta and high < 1e12:
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (lower + high)
        if np.linalg.norm(ph(mid)) > delta:
            lower = mid
        else:
            high = mid
    p = Q @ ph(high)
    n = np.linalg.norm(p)
    if n > delta * (1.0 + 1e-8):
        p *= delta / n
    return p


def quadratic_change(z: np.ndarray, p: np.ndarray, b: np.ndarray, H: np.ndarray) -> float:
    return float((b + H @ z) @ p + 0.5 * p @ (H @ p))


def initial_map(
    oracle: NeuralOracle,
    anchor: torch.Tensor,
    f0: float,
    g: torch.Tensor,
    V: torch.Tensor,
    e: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, float]]]:
    b = (V.T @ g).detach().cpu().numpy().astype(float)
    z1 = np.array([e, 0.0])
    z2 = np.array([0.0, e])
    z3 = np.array([e, e])

    def theta_at(z: np.ndarray) -> torch.Tensor:
        zt = torch.tensor(z, device=anchor.device, dtype=anchor.dtype)
        return anchor + V @ zt

    f1 = oracle.value(theta_at(z1))
    f2 = oracle.value(theta_at(z2))
    f3 = oracle.value(theta_at(z3))
    h11 = 2.0 * ((f1 - f0) - b[0] * e) / (e * e)
    h22 = 2.0 * ((f2 - f0) - b[1] * e) / (e * e)
    residual = (f3 - f0) - b @ z3 - 0.5 * h11 * e * e - 0.5 * h22 * e * e
    h12 = residual / (e * e)
    H = np.array([[h11, h12], [h12, h22]], dtype=float)
    obs = [(z1, f1 - f0), (z2, f2 - f0), (z3, f3 - f0)]
    return b, H, obs


def refit_H(
    b: np.ndarray,
    obs: List[Tuple[np.ndarray, float]],
    H_prior: np.ndarray,
    z_current: np.ndarray,
    scale: float,
    ridge: float = 1e-8,
) -> np.ndarray:
    if len(obs) < 3:
        return H_prior
    X, y, w = [], [], []
    sigma = max(scale, 1e-8)
    for z, dy in obs:
        z1, z2 = float(z[0]), float(z[1])
        X.append([0.5 * z1 * z1, z1 * z2, 0.5 * z2 * z2])
        y.append(float(dy - b @ z))
        dist = np.linalg.norm(z - z_current)
        w.append(math.exp(-0.5 * (dist / sigma) ** 2))
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w)
    hp = np.array([H_prior[0, 0], H_prior[0, 1], H_prior[1, 1]])
    sw = np.sqrt(np.maximum(w, 1e-8))
    Xw = X * sw[:, None]
    yw = y * sw
    A = Xw.T @ Xw + ridge * np.eye(3)
    rhs = Xw.T @ yw + ridge * hp
    try:
        h = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        h = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.array([[h[0], h[1]], [h[1], h[2]]], dtype=float)


def refit_H_probe_free(
    b: np.ndarray,
    obs: List[Tuple[np.ndarray, float]],
    H_prior: np.ndarray,
    z_current: np.ndarray,
    scale: float,
    prior_strength: float = 1e-2,
) -> np.ndarray:
    """
    Fit the 2x2 curvature matrix from whatever accepted/trial observations are
    already available. Unlike refit_H(), this works with one or two
    observations by regularizing toward H_prior. No dedicated probe calls are
    required.
    """
    if not obs:
        return H_prior.copy()
    X, y, w = [], [], []
    sigma = max(scale, 1e-12)
    for z, dy in obs:
        z1, z2 = float(z[0]), float(z[1])
        X.append([0.5 * z1 * z1, z1 * z2, 0.5 * z2 * z2])
        y.append(float(dy - b @ z))
        dist = np.linalg.norm(z - z_current)
        w.append(math.exp(-0.5 * (dist / sigma) ** 2))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    hp = np.array([H_prior[0, 0], H_prior[0, 1], H_prior[1, 1]], dtype=float)
    sw = np.sqrt(np.maximum(w, 1e-8))
    Xw = X * sw[:, None]
    yw = y * sw
    XtX = Xw.T @ Xw
    data_scale = max(float(np.trace(XtX)) / 3.0, 1e-24)
    lam = prior_strength * data_scale
    A = XtX + lam * np.eye(3)
    rhs = Xw.T @ yw + lam * hp
    try:
        h = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        h = np.linalg.lstsq(A, rhs, rcond=None)[0]
    H = np.array([[h[0], h[1]], [h[1], h[2]]], dtype=float)
    if not np.all(np.isfinite(H)):
        return H_prior.copy()
    return 0.5 * (H + H.T)


def transferred_curvature_prior(
    theta: torch.Tensor,
    g: torch.Tensor,
    V_new: torch.Tensor,
    delta: float,
    old_anchor: Optional[torch.Tensor],
    old_g: Optional[torch.Tensor],
    old_V: Optional[torch.Tensor],
    old_H: Optional[np.ndarray],
) -> np.ndarray:
    """Build a no-probe curvature prior in the new 2-D subspace."""
    b_new = (V_new.T @ g).detach().cpu().numpy().astype(float)
    lam_default = max(float(np.linalg.norm(b_new)) / max(delta, 1e-12), 1e-8)
    lam = lam_default
    if old_anchor is not None and old_g is not None:
        s = theta - old_anchor
        y = g - old_g
        ss = float(torch.dot(s, s))
        sy = float(torch.dot(s, y))
        if ss > 1e-20 and np.isfinite(sy):
            bb = abs(sy) / ss
            if np.isfinite(bb) and bb > 1e-12:
                lam = float(np.clip(bb, 0.05 * lam_default, 20.0 * lam_default))

    H = lam * np.eye(2, dtype=float)
    if old_V is not None and old_H is not None:
        M = (old_V.T @ V_new).detach().cpu().numpy().astype(float)
        overlap = M.T @ M
        Hproj = M.T @ old_H @ M
        H = Hproj + lam * (np.eye(2) - overlap)
        H = 0.5 * (H + H.T)
        if not np.all(np.isfinite(H)):
            H = lam * np.eye(2, dtype=float)
    return H


# ---------------------------------------------------------------------------
# AGRLS neural-network diagnostic
# ---------------------------------------------------------------------------

@dataclass
class AGRLSNNConfig:
    max_without_grad: int = 3
    agreement_good: float = 2.0 / 3.0
    agreement_ok: float = 0.25
    grow: float = 1.5
    shrink: float = 0.5
    moderate_shrink: float = 0.8
    max_rejects: int = 2
    probe_target_rel_loss: float = 0.005
    probe_min_rel_theta: float = 1e-6
    probe_max_rel_theta: float = 5e-3
    trust_initial_mult: float = 4.0
    trust_max_mult: float = 32.0


def auto_probe_radius(theta: torch.Tensor, f: float, g: torch.Tensor, cfg: AGRLSNNConfig) -> float:
    gnorm = float(torch.linalg.vector_norm(g))
    tnorm = max(float(torch.linalg.vector_norm(theta)), 1.0)
    raw = cfg.probe_target_rel_loss * max(abs(f), 0.1) / max(gnorm, 1e-12)
    lo = cfg.probe_min_rel_theta * tnorm
    hi = cfg.probe_max_rel_theta * tnorm
    return float(np.clip(raw, lo, hi))


def run_agrls(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    seed: int,
    updates: int,
    policy: str,
    measured_r: float,
    cfg: AGRLSNNConfig,
) -> Tuple[List[dict], dict]:
    model.load_state_dict(init_state)
    model.to(device)
    oracle = NeuralOracle(model, xtr, ytr, loss_fn, device)
    theta = params_to_vector(model).detach().clone()
    gen = torch.Generator(device=device).manual_seed(seed + 91001)

    prev_step = None
    need_refresh = True
    refresh_reason = "initial"
    accepted_updates = 0
    since_refresh = 0
    attempts = 0
    consecutive_rejects = 0
    cycle_first_gain = None
    rows: List[dict] = []

    anchor = theta.clone()
    anchor_f = float("nan")
    g_anchor = torch.zeros_like(theta)
    V = None
    b = None
    H = None
    z_current = np.zeros(2)
    obs: List[Tuple[np.ndarray, float]] = []
    e = 1e-3
    delta = None
    delta_max = None

    sync(device)
    run_t0 = time.perf_counter()

    while accepted_updates < updates:
        if need_refresh:
            f_current, g = oracle.value_grad(theta)
            anchor = theta.clone()
            anchor_f = f_current
            g_anchor = g.clone()
            V = orthonormal_basis(g, prev_step, gen)
            z_current = np.zeros(2)
            e = auto_probe_radius(theta, f_current, g, cfg)
            if delta is None:
                delta = cfg.trust_initial_mult * e
            else:
                delta = float(np.clip(delta, 2.0 * e, cfg.trust_max_mult * e))
            delta_max = cfg.trust_max_mult * e
            b, H, obs = initial_map(oracle, anchor, f_current, g, V, e)
            since_refresh = 0
            consecutive_rejects = 0
            cycle_first_gain = None
            need_refresh = False
            rows.append({
                "algorithm": f"agrls_{policy}", "update": accepted_updates,
                "attempt": attempts, "event": "gradient_refresh_and_map",
                "refresh_reason": refresh_reason, "accepted": 0,
                "objective": f_current, "forward_calls": oracle.forward_calls,
                "gradient_calls": oracle.gradient_calls,
                "forward_seconds": oracle.forward_seconds,
                "gradient_seconds": oracle.gradient_seconds,
                "trust_radius": delta, "probe_radius": e,
                "since_refresh": 0, "predicted_improvement": float("nan"),
                "actual_improvement": float("nan"), "rho": float("nan"),
                "agreement": float("nan"),
            })

        assert V is not None and b is not None and H is not None and delta is not None
        local_g = b + H @ z_current
        p = solve_full_trust_region(local_g, H, delta)
        predicted = -quadratic_change(z_current, p, b, H)

        if policy == "mvr" and since_refresh >= 1 and cycle_first_gain is not None and predicted > 0 and np.isfinite(predicted):
            threshold = 1.0 / (measured_r + 4.0)
            continuation_ratio = predicted / max(cycle_first_gain, 1e-15)
            if continuation_ratio < threshold:
                need_refresh = True
                refresh_reason = "marginal_value_below_refresh_value"
                attempts += 1
                rows.append({
                    "algorithm": f"agrls_{policy}", "update": accepted_updates,
                    "attempt": attempts, "event": "mvr_refresh_decision",
                    "refresh_reason": refresh_reason, "accepted": 0,
                    "objective": f_current, "forward_calls": oracle.forward_calls,
                    "gradient_calls": oracle.gradient_calls,
                    "forward_seconds": oracle.forward_seconds,
                    "gradient_seconds": oracle.gradient_seconds,
                    "trust_radius": delta, "probe_radius": e,
                    "since_refresh": since_refresh,
                    "predicted_improvement": predicted,
                    "actual_improvement": float("nan"), "rho": float("nan"),
                    "agreement": float("nan"),
                    "continuation_value_ratio": continuation_ratio,
                    "mvr_threshold": threshold,
                })
                continue

        if (not np.isfinite(predicted)) or predicted <= 1e-12 or np.linalg.norm(p) <= 1e-12:
            need_refresh = True
            refresh_reason = "subspace_exhausted"
            attempts += 1
            continue

        z_trial = z_current + p
        p_t = torch.tensor(p, device=device, dtype=theta.dtype)
        theta_trial = anchor + V @ torch.tensor(z_trial, device=device, dtype=theta.dtype)
        f_trial = oracle.value(theta_trial)
        obs.append((z_trial.copy(), f_trial - anchor_f))
        actual = f_current - f_trial
        rho = actual / max(predicted, 1e-15)
        agreement = min(rho, 1.0 / rho) if rho > 0 and np.isfinite(rho) else 0.0

        accepted = False
        event = "reject_model_miss"
        if actual > 0 and agreement >= cfg.agreement_good:
            accepted = True
            event = "accept_good"
            delta = min(delta_max, cfg.grow * delta)
            consecutive_rejects = 0
        elif actual > 0 and agreement >= cfg.agreement_ok:
            accepted = True
            event = "accept_ok"
            delta = max(0.5 * e, cfg.moderate_shrink * delta)
            consecutive_rejects = 0
        elif actual > 0:
            accepted = True
            event = "accept_stale_model"
            delta = max(0.5 * e, cfg.shrink * delta)
            consecutive_rejects = 0
        else:
            consecutive_rejects += 1
            delta = max(0.5 * e, cfg.shrink * delta)

        fit_center = z_trial if accepted else z_current
        H = refit_H(b, obs, H, fit_center, scale=max(2.0 * delta, 4.0 * e))
        attempts += 1

        if accepted:
            theta_old = theta
            theta = theta_trial
            f_current = f_trial
            z_current = z_trial
            prev_step = theta - theta_old
            accepted_updates += 1
            since_refresh += 1
            if since_refresh == 1:
                cycle_first_gain = max(float(actual), 1e-15)
            if policy == "fixed1":
                need_refresh = True
                refresh_reason = "fixed_1_move"
            elif since_refresh >= cfg.max_without_grad:
                need_refresh = True
                refresh_reason = "mvr_safety_k3"
        elif consecutive_rejects >= cfg.max_rejects:
            need_refresh = True
            refresh_reason = "repeated_model_miss"

        rows.append({
            "algorithm": f"agrls_{policy}", "update": accepted_updates,
            "attempt": attempts, "event": event,
            "refresh_reason": refresh_reason if need_refresh else "",
            "accepted": int(accepted), "objective": f_current,
            "trial_objective": f_trial,
            "forward_calls": oracle.forward_calls,
            "gradient_calls": oracle.gradient_calls,
            "forward_seconds": oracle.forward_seconds,
            "gradient_seconds": oracle.gradient_seconds,
            "trust_radius": delta, "probe_radius": e,
            "since_refresh": since_refresh,
            "predicted_improvement": predicted,
            "actual_improvement": actual, "rho": rho,
            "agreement": agreement,
            "continuation_value_ratio": (predicted / max(cycle_first_gain, 1e-15)) if cycle_first_gain else float("nan"),
            "mvr_threshold": (1.0 / (measured_r + 4.0)) if policy == "mvr" else float("nan"),
        })

        if attempts > updates * 40:
            raise RuntimeError("AGRLS exceeded attempt safety limit")

    copy_vector_to_params(model, theta)
    sync(device)
    elapsed = time.perf_counter() - run_t0
    train_loss, train_acc = evaluate(model, xtr, ytr, loss_fn)
    eval_loss, eval_acc = evaluate(model, xev, yev, loss_fn)
    summary = {
        "algorithm": f"agrls_{policy}",
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "final_eval_loss": eval_loss,
        "final_eval_accuracy": eval_acc,
        "accepted_updates": accepted_updates,
        "forward_calls": oracle.forward_calls,
        "gradient_calls": oracle.gradient_calls,
        "forward_seconds": oracle.forward_seconds,
        "gradient_seconds": oracle.gradient_seconds,
        "oracle_seconds": oracle.forward_seconds + oracle.gradient_seconds,
        "wall_seconds": elapsed,
        "measured_gradient_cost_ratio": measured_r,
        "max_moves_between_gradients": max((int(r.get("since_refresh", 0)) for r in rows), default=0),
    }
    return rows, summary


def run_agrls_probe_free(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    seed: int,
    updates: int,
    measured_r: float,
    cfg: AGRLSNNConfig,
) -> Tuple[List[dict], dict]:
    """
    Probe-free AGRLS-MVR. Dedicated map probes are eliminated.

    At each exact-gradient refresh:
      * the new linear term is exact: b = V^T g;
      * curvature is transferred from the previous subspace plus a
        Barzilai-Borwein-like scalar prior;
      * every mandatory trial forward evaluation becomes a curvature
        observation for online surrogate repair.
    """
    model.load_state_dict(init_state)
    model.to(device)
    oracle = NeuralOracle(model, xtr, ytr, loss_fn, device)
    theta = params_to_vector(model).detach().clone()
    gen = torch.Generator(device=device).manual_seed(seed + 191001)

    prev_step = None
    need_refresh = True
    refresh_reason = "initial"
    accepted_updates = 0
    since_refresh = 0
    attempts = 0
    consecutive_rejects = 0
    cycle_first_gain = None
    rows: List[dict] = []

    anchor = theta.clone()
    anchor_f = float("nan")
    g_anchor = torch.zeros_like(theta)
    V = None
    b = None
    H = None
    H_prior = None
    z_current = np.zeros(2)
    obs: List[Tuple[np.ndarray, float]] = []
    e = 1e-3
    delta = None
    delta_max = None

    sync(device)
    run_t0 = time.perf_counter()

    while accepted_updates < updates:
        if need_refresh:
            old_anchor = anchor.clone() if V is not None else None
            old_g = g_anchor.clone() if V is not None else None
            old_V = V.clone() if V is not None else None
            old_H = H.copy() if H is not None else None

            f_current, g = oracle.value_grad(theta)
            anchor = theta.clone()
            anchor_f = f_current
            g_anchor = g.clone()
            V = orthonormal_basis(g, prev_step, gen)
            z_current = np.zeros(2)
            e = auto_probe_radius(theta, f_current, g, cfg)
            if delta is None:
                delta = cfg.trust_initial_mult * e
            else:
                delta = float(np.clip(delta, 2.0 * e, cfg.trust_max_mult * e))
            delta_max = cfg.trust_max_mult * e
            b = (V.T @ g).detach().cpu().numpy().astype(float)
            H_prior = transferred_curvature_prior(
                theta, g, V, delta, old_anchor, old_g, old_V, old_H
            )
            H = H_prior.copy()
            obs = []
            since_refresh = 0
            consecutive_rejects = 0
            cycle_first_gain = None
            need_refresh = False
            rows.append({
                "algorithm": "agrls_pf_mvr", "update": accepted_updates,
                "attempt": attempts, "event": "gradient_refresh_probe_free",
                "refresh_reason": refresh_reason, "accepted": 0,
                "objective": f_current, "forward_calls": oracle.forward_calls,
                "gradient_calls": oracle.gradient_calls,
                "forward_seconds": oracle.forward_seconds,
                "gradient_seconds": oracle.gradient_seconds,
                "trust_radius": delta, "scale_radius": e,
                "since_refresh": 0, "predicted_improvement": float("nan"),
                "actual_improvement": float("nan"), "rho": float("nan"),
                "agreement": float("nan"),
            })

        assert V is not None and b is not None and H is not None and H_prior is not None and delta is not None
        local_g = b + H @ z_current
        p = solve_full_trust_region(local_g, H, delta)
        predicted = -quadratic_change(z_current, p, b, H)

        if since_refresh >= 1 and cycle_first_gain is not None and predicted > 0 and np.isfinite(predicted):
            # Refresh now costs one gradient query plus the next trial forward.
            threshold = 1.0 / (measured_r + 1.0)
            continuation_ratio = predicted / max(cycle_first_gain, 1e-15)
            if continuation_ratio < threshold:
                need_refresh = True
                refresh_reason = "pf_marginal_value_below_refresh_value"
                attempts += 1
                rows.append({
                    "algorithm": "agrls_pf_mvr", "update": accepted_updates,
                    "attempt": attempts, "event": "mvr_refresh_decision",
                    "refresh_reason": refresh_reason, "accepted": 0,
                    "objective": f_current, "forward_calls": oracle.forward_calls,
                    "gradient_calls": oracle.gradient_calls,
                    "forward_seconds": oracle.forward_seconds,
                    "gradient_seconds": oracle.gradient_seconds,
                    "trust_radius": delta, "scale_radius": e,
                    "since_refresh": since_refresh,
                    "predicted_improvement": predicted,
                    "actual_improvement": float("nan"), "rho": float("nan"),
                    "agreement": float("nan"),
                    "continuation_value_ratio": continuation_ratio,
                    "mvr_threshold": threshold,
                })
                continue

        if (not np.isfinite(predicted)) or predicted <= 1e-12 or np.linalg.norm(p) <= 1e-12:
            need_refresh = True
            refresh_reason = "subspace_exhausted"
            attempts += 1
            continue

        z_trial = z_current + p
        theta_trial = anchor + V @ torch.tensor(z_trial, device=device, dtype=theta.dtype)
        f_trial = oracle.value(theta_trial)
        obs.append((z_trial.copy(), f_trial - anchor_f))
        actual = f_current - f_trial
        rho = actual / max(predicted, 1e-15)
        agreement = min(rho, 1.0 / rho) if rho > 0 and np.isfinite(rho) else 0.0

        accepted = False
        event = "reject_model_miss"
        if actual > 0 and agreement >= cfg.agreement_good:
            accepted = True
            event = "accept_good"
            delta = min(delta_max, cfg.grow * delta)
            consecutive_rejects = 0
        elif actual > 0 and agreement >= cfg.agreement_ok:
            accepted = True
            event = "accept_ok"
            delta = max(0.5 * e, cfg.moderate_shrink * delta)
            consecutive_rejects = 0
        elif actual > 0:
            accepted = True
            event = "accept_stale_model"
            delta = max(0.5 * e, cfg.shrink * delta)
            consecutive_rejects = 0
        else:
            consecutive_rejects += 1
            delta = max(0.5 * e, cfg.shrink * delta)

        fit_center = z_trial if accepted else z_current
        H = refit_H_probe_free(
            b, obs, H_prior, fit_center, scale=max(2.0 * delta, 4.0 * e)
        )
        attempts += 1

        if accepted:
            theta_old = theta
            theta = theta_trial
            f_current = f_trial
            z_current = z_trial
            prev_step = theta - theta_old
            accepted_updates += 1
            since_refresh += 1
            if since_refresh == 1:
                cycle_first_gain = max(float(actual), 1e-15)
            if since_refresh >= cfg.max_without_grad:
                need_refresh = True
                refresh_reason = "pf_safety_cap"
        elif consecutive_rejects >= cfg.max_rejects:
            need_refresh = True
            refresh_reason = "repeated_model_miss"

        rows.append({
            "algorithm": "agrls_pf_mvr", "update": accepted_updates,
            "attempt": attempts, "event": event,
            "refresh_reason": refresh_reason if need_refresh else "",
            "accepted": int(accepted), "objective": f_current,
            "trial_objective": f_trial,
            "forward_calls": oracle.forward_calls,
            "gradient_calls": oracle.gradient_calls,
            "forward_seconds": oracle.forward_seconds,
            "gradient_seconds": oracle.gradient_seconds,
            "trust_radius": delta, "scale_radius": e,
            "since_refresh": since_refresh,
            "predicted_improvement": predicted,
            "actual_improvement": actual, "rho": rho,
            "agreement": agreement,
            "continuation_value_ratio": (predicted / max(cycle_first_gain, 1e-15)) if cycle_first_gain else float("nan"),
            "mvr_threshold": (1.0 / (measured_r + 1.0)),
        })

        if attempts > updates * 60:
            raise RuntimeError("Probe-free AGRLS exceeded attempt safety limit")

    copy_vector_to_params(model, theta)
    sync(device)
    elapsed = time.perf_counter() - run_t0
    train_loss, train_acc = evaluate(model, xtr, ytr, loss_fn)
    eval_loss, eval_acc = evaluate(model, xev, yev, loss_fn)
    summary = {
        "algorithm": "agrls_pf_mvr",
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "final_eval_loss": eval_loss,
        "final_eval_accuracy": eval_acc,
        "accepted_updates": accepted_updates,
        "forward_calls": oracle.forward_calls,
        "gradient_calls": oracle.gradient_calls,
        "forward_seconds": oracle.forward_seconds,
        "gradient_seconds": oracle.gradient_seconds,
        "oracle_seconds": oracle.forward_seconds + oracle.gradient_seconds,
        "wall_seconds": elapsed,
        "measured_gradient_cost_ratio": measured_r,
        "max_moves_between_gradients": max((int(r.get("since_refresh", 0)) for r in rows), default=0),
        "dedicated_map_probe_calls": 0,
    }
    return rows, summary



def adam_preconditioned_basis(
    g: torch.Tensor,
    adam_direction: torch.Tensor,
    prev_step: Optional[torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    """2-D basis led by the Adam-preconditioned descent direction.

    v1 follows the coordinate-preconditioned Adam direction. v2 tries to
    retain raw steepest-descent information orthogonal to v1, then falls back
    to the previous accepted step or a random orthogonal direction.
    """
    eps = 1e-15
    nd = torch.linalg.vector_norm(adam_direction)
    if float(nd) > eps:
        v1 = adam_direction / nd
    else:
        ng = torch.linalg.vector_norm(g)
        v1 = (-g / ng) if float(ng) > eps else torch.randn(
            g.numel(), device=g.device, dtype=g.dtype, generator=generator
        )
        v1 = v1 / torch.linalg.vector_norm(v1)

    candidates = [-g]
    if prev_step is not None:
        candidates.append(prev_step)
    v2 = None
    for cand in candidates:
        if cand is None or float(torch.linalg.vector_norm(cand)) <= eps:
            continue
        tmp = cand - torch.dot(cand, v1) * v1
        n = torch.linalg.vector_norm(tmp)
        if float(n) > 1e-10:
            v2 = tmp / n
            break
    if v2 is None:
        for _ in range(20):
            cand = torch.randn(g.numel(), device=g.device, dtype=g.dtype, generator=generator)
            cand = cand - torch.dot(cand, v1) * v1
            n = torch.linalg.vector_norm(cand)
            if float(n) > 1e-10:
                v2 = cand / n
                break
    if v2 is None:
        raise RuntimeError('Could not construct Adam-preconditioned 2-D basis')
    return torch.stack([v1, v2], dim=1)


def run_agrls_adam_probe_free(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    seed: int,
    updates: int,
    measured_r: float,
    cfg: AGRLSNNConfig,
    adam_lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> Tuple[List[dict], dict]:
    """Adam-anchored, probe-free AGRLS-MVR.

    Each expensive gradient refresh updates Adam moments. The first candidate
    move after the refresh is the ordinary Adam-preconditioned step (represented
    exactly inside the local 2-D basis). Subsequent accepted moves reuse the
    same gradient/subspace and learn curvature only from mandatory trial forward
    evaluations. This isolates whether coordinate preconditioning can recover
    the optimization quality lost by the raw-gradient probe-free variant.
    """
    model.load_state_dict(init_state)
    model.to(device)
    oracle = NeuralOracle(model, xtr, ytr, loss_fn, device)
    theta = params_to_vector(model).detach().clone()
    gen = torch.Generator(device=device).manual_seed(seed + 271001)

    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    adam_t = 0

    prev_step = None
    need_refresh = True
    refresh_reason = 'initial'
    accepted_updates = 0
    since_refresh = 0
    attempts = 0
    consecutive_rejects = 0
    cycle_first_gain = None
    rows: List[dict] = []

    anchor = theta.clone()
    anchor_f = float('nan')
    g_anchor = torch.zeros_like(theta)
    V = None
    b = None
    H = None
    H_prior = None
    z_current = np.zeros(2)
    obs: List[Tuple[np.ndarray, float]] = []
    e = 1e-3
    delta = None
    delta_max = None
    first_move_pending = False
    first_step_norm = 0.0

    sync(device)
    run_t0 = time.perf_counter()

    while accepted_updates < updates:
        if need_refresh:
            old_anchor = anchor.clone() if V is not None else None
            old_g = g_anchor.clone() if V is not None else None
            old_V = V.clone() if V is not None else None
            old_H = H.copy() if H is not None else None

            f_current, g = oracle.value_grad(theta)
            adam_t += 1
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            mhat = m / (1.0 - beta1 ** adam_t)
            vhat = v / (1.0 - beta2 ** adam_t)
            adam_direction = -mhat / (torch.sqrt(vhat) + adam_eps)
            adam_step = adam_lr * adam_direction
            step_norm_tensor = torch.linalg.vector_norm(adam_step)
            first_step_norm = max(float(step_norm_tensor), 1e-12)

            anchor = theta.clone()
            anchor_f = f_current
            g_anchor = g.clone()
            V = adam_preconditioned_basis(g, adam_direction, prev_step, gen)
            z_current = np.zeros(2)
            e_auto = auto_probe_radius(theta, f_current, g, cfg)
            e = max(e_auto, 0.10 * first_step_norm)
            delta = max(first_step_norm, 2.0 * e)
            delta_max = max(4.0 * first_step_norm, cfg.trust_max_mult * e)
            b = (V.T @ g).detach().cpu().numpy().astype(float)
            H_prior = transferred_curvature_prior(
                theta, g, V, delta, old_anchor, old_g, old_V, old_H
            )
            H = H_prior.copy()
            obs = []
            since_refresh = 0
            consecutive_rejects = 0
            cycle_first_gain = None
            first_move_pending = True
            anchor_backtracks = 0
            need_refresh = False
            rows.append({
                'algorithm': 'agrls_adam_pf_mvr', 'update': accepted_updates,
                'attempt': attempts, 'event': 'gradient_refresh_adam_anchor',
                'refresh_reason': refresh_reason, 'accepted': 0,
                'objective': f_current, 'forward_calls': oracle.forward_calls,
                'gradient_calls': oracle.gradient_calls,
                'forward_seconds': oracle.forward_seconds,
                'gradient_seconds': oracle.gradient_seconds,
                'trust_radius': delta, 'scale_radius': e,
                'adam_step_norm': first_step_norm, 'since_refresh': 0,
                'predicted_improvement': float('nan'), 'actual_improvement': float('nan'),
                'rho': float('nan'), 'agreement': float('nan'),
            })

        assert V is not None and b is not None and H is not None and H_prior is not None

        if first_move_pending:
            p = np.array([first_step_norm, 0.0], dtype=float)
            predicted = float(-(b @ p))  # first-order prediction only
        else:
            local_g = b + H @ z_current
            p = solve_full_trust_region(local_g, H, delta)
            predicted = -quadratic_change(z_current, p, b, H)

            if since_refresh >= 1 and cycle_first_gain is not None and predicted > 0 and np.isfinite(predicted):
                threshold = 1.0 / (measured_r + 1.0)
                continuation_ratio = predicted / max(cycle_first_gain, 1e-15)
                if continuation_ratio < threshold:
                    need_refresh = True
                    refresh_reason = 'adam_pf_marginal_value_below_refresh_value'
                    attempts += 1
                    rows.append({
                        'algorithm': 'agrls_adam_pf_mvr', 'update': accepted_updates,
                        'attempt': attempts, 'event': 'mvr_refresh_decision',
                        'refresh_reason': refresh_reason, 'accepted': 0,
                        'objective': f_current, 'forward_calls': oracle.forward_calls,
                        'gradient_calls': oracle.gradient_calls,
                        'forward_seconds': oracle.forward_seconds, 'gradient_seconds': oracle.gradient_seconds,
                        'trust_radius': delta, 'scale_radius': e, 'since_refresh': since_refresh,
                        'predicted_improvement': predicted, 'actual_improvement': float('nan'),
                        'rho': float('nan'), 'agreement': float('nan'),
                        'continuation_value_ratio': continuation_ratio, 'mvr_threshold': threshold,
                    })
                    continue

        if (not np.isfinite(predicted)) or predicted <= 1e-12 or np.linalg.norm(p) <= 1e-12:
            need_refresh = True
            refresh_reason = 'subspace_exhausted'
            attempts += 1
            continue

        z_trial = z_current + p
        theta_trial = anchor + V @ torch.tensor(z_trial, device=device, dtype=theta.dtype)
        f_trial = oracle.value(theta_trial)
        obs.append((z_trial.copy(), f_trial - anchor_f))
        actual = f_current - f_trial
        rho = actual / max(predicted, 1e-15)
        agreement = min(rho, 1.0 / rho) if rho > 0 and np.isfinite(rho) else 0.0

        accepted = False
        event = 'reject_model_miss'
        if actual > 0 and agreement >= cfg.agreement_good:
            accepted = True; event = 'accept_good'
            delta = min(delta_max, cfg.grow * delta); consecutive_rejects = 0
        elif actual > 0 and agreement >= cfg.agreement_ok:
            accepted = True; event = 'accept_ok'
            delta = max(0.25 * e, cfg.moderate_shrink * delta); consecutive_rejects = 0
        elif actual > 0:
            accepted = True; event = 'accept_stale_model'
            delta = max(0.25 * e, cfg.shrink * delta); consecutive_rejects = 0
        else:
            consecutive_rejects += 1
            delta = max(0.25 * e, cfg.shrink * delta)
            if first_move_pending:
                first_step_norm *= 0.5
                event = 'reject_adam_anchor_backtrack'

        fit_center = z_trial if accepted else z_current
        H = refit_H_probe_free(b, obs, H_prior, fit_center, scale=max(2.0 * delta, 4.0 * e))
        attempts += 1

        if accepted:
            theta_old = theta
            theta = theta_trial
            f_current = f_trial
            z_current = z_trial
            prev_step = theta - theta_old
            accepted_updates += 1
            since_refresh += 1
            if since_refresh == 1:
                cycle_first_gain = max(float(actual), 1e-15)
            first_move_pending = False
            if since_refresh >= cfg.max_without_grad:
                need_refresh = True
                refresh_reason = 'adam_pf_safety_cap'
        elif consecutive_rejects >= 3:
            need_refresh = True
            refresh_reason = 'repeated_model_miss'

        rows.append({
            'algorithm': 'agrls_adam_pf_mvr', 'update': accepted_updates,
            'attempt': attempts, 'event': event,
            'refresh_reason': refresh_reason if need_refresh else '',
            'accepted': int(accepted), 'objective': f_current, 'trial_objective': f_trial,
            'forward_calls': oracle.forward_calls, 'gradient_calls': oracle.gradient_calls,
            'forward_seconds': oracle.forward_seconds, 'gradient_seconds': oracle.gradient_seconds,
            'trust_radius': delta, 'scale_radius': e, 'adam_step_norm': first_step_norm,
            'since_refresh': since_refresh, 'predicted_improvement': predicted,
            'actual_improvement': actual, 'rho': rho, 'agreement': agreement,
            'continuation_value_ratio': (predicted / max(cycle_first_gain, 1e-15)) if cycle_first_gain else float('nan'),
            'mvr_threshold': (1.0 / (measured_r + 1.0)),
        })

        if attempts > updates * 80:
            raise RuntimeError('Adam-anchored probe-free AGRLS exceeded attempt safety limit')

    copy_vector_to_params(model, theta)
    sync(device)
    elapsed = time.perf_counter() - run_t0
    train_loss, train_acc = evaluate(model, xtr, ytr, loss_fn)
    eval_loss, eval_acc = evaluate(model, xev, yev, loss_fn)
    summary = {
        'algorithm': 'agrls_adam_pf_mvr', 'final_train_loss': train_loss,
        'final_train_accuracy': train_acc, 'final_eval_loss': eval_loss,
        'final_eval_accuracy': eval_acc, 'accepted_updates': accepted_updates,
        'forward_calls': oracle.forward_calls, 'gradient_calls': oracle.gradient_calls,
        'forward_seconds': oracle.forward_seconds, 'gradient_seconds': oracle.gradient_seconds,
        'oracle_seconds': oracle.forward_seconds + oracle.gradient_seconds,
        'wall_seconds': elapsed, 'measured_gradient_cost_ratio': measured_r,
        'max_moves_between_gradients': max((int(r.get('since_refresh', 0)) for r in rows), default=0),
        'dedicated_map_probe_calls': 0, 'adam_anchor_lr': adam_lr,
    }
    return rows, summary


def refit_h_1d(
    b: float,
    obs: List[Tuple[float, float]],
    h_prior: float,
    z_center: float,
    scale: float,
    ridge: float = 1e-8,
) -> float:
    """Fit the single 1-D quadratic curvature coefficient.

    With exact directional derivative b known at the anchor, the 1-D model
        Delta f(z) = b z + 0.5 h z^2
    has only one unknown curvature coefficient h. One nonzero forward
    observation is therefore sufficient to identify h (up to regularization).
    """
    num = ridge * float(h_prior)
    den = ridge
    scl = max(float(scale), 1e-12)
    for z, y in obs:
        z = float(z)
        a = 0.5 * z * z
        if abs(a) <= 1e-18:
            continue
        target = float(y) - float(b) * z
        w = 1.0 / (1.0 + abs(z - float(z_center)) / scl)
        num += w * a * target
        den += w * a * a
    h = num / max(den, 1e-18)
    return float(np.clip(h, -1e8, 1e8))


def run_agrls_adam_probe_free_1d(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    seed: int,
    updates: int,
    measured_r: float,
    cfg: AGRLSNNConfig,
    adam_lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> Tuple[List[dict], dict]:
    """AGRLS-v0.8.1: Adam-anchored, information-matched 1-D PF-MVR.

    Rationale: after an exact gradient the linear term is known. A full 2-D
    quadratic still contains three unknown curvature coefficients, but the
    first mandatory forward trial supplies only one scalar curvature
    observation. The v0.7 2-D model is therefore underdetermined immediately
    after refresh. v0.8 reduces the reusable local model to one dimension, so
    that the first accepted Adam-direction trial exactly identifies the sole
    quadratic curvature coefficient. Further forward-only trials refine that
    coefficient online.

    Rejected continuation trials are bounded by a hardware-cost-aware budget:
    at most floor(measured_r) forward misses are paid before refreshing the
    gradient, because beyond that point another exact gradient is no more
    expensive under the measured oracle-cost model.
    """
    model.load_state_dict(init_state)
    model.to(device)
    oracle = NeuralOracle(model, xtr, ytr, loss_fn, device)
    theta = params_to_vector(model).detach().clone()

    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    adam_t = 0

    need_refresh = True
    refresh_reason = 'initial'
    accepted_updates = 0
    since_refresh = 0
    attempts = 0
    consecutive_rejects = 0
    cycle_first_gain = None
    rows: List[dict] = []

    anchor = theta.clone()
    anchor_f = float('nan')
    g_anchor = torch.zeros_like(theta)
    v1 = torch.zeros_like(theta)
    b1 = 0.0
    h1 = 0.0
    h_prior = 0.0
    z_current = 0.0
    obs: List[Tuple[float, float]] = []
    delta = 0.0
    delta_max = 0.0
    first_move_pending = False
    first_step_norm = 0.0
    old_refresh_theta: Optional[torch.Tensor] = None
    old_refresh_grad: Optional[torch.Tensor] = None
    retry_budget = max(1, int(math.floor(max(measured_r, 1.0))))
    # v0.8.1 bug fix: an Adam-anchor rejection occurs at the same theta used
    # for the exact gradient. Refreshing the gradient at that unchanged point
    # is redundant and can reset the full Adam step forever when retry_budget=1.
    # Anchor backtracking therefore reuses the same gradient and halves only the
    # step length. The cost-aware retry budget remains active for continuation
    # misses after at least one accepted move, where a new gradient is genuinely
    # informative at a new theta.
    max_anchor_backtracks = 12
    anchor_backtracks = 0
    anchor_direction_fallbacks = 0

    sync(device)
    run_t0 = time.perf_counter()

    while accepted_updates < updates:
        if need_refresh:
            prev_refresh_theta = old_refresh_theta
            prev_refresh_grad = old_refresh_grad

            f_current, g = oracle.value_grad(theta)
            adam_t += 1
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            mhat = m / (1.0 - beta1 ** adam_t)
            vhat = v / (1.0 - beta2 ** adam_t)
            adam_direction = -mhat / (torch.sqrt(vhat) + adam_eps)
            nd = torch.linalg.vector_norm(adam_direction)
            if float(nd) <= 1e-15:
                ng = torch.linalg.vector_norm(g)
                if float(ng) <= 1e-15:
                    break
                v1 = -g / ng
                first_step_norm = max(adam_lr * float(ng), 1e-12)
            else:
                v1 = adam_direction / nd
                first_step_norm = max(adam_lr * float(nd), 1e-12)

            anchor = theta.clone()
            anchor_f = f_current
            g_anchor = g.clone()
            b1 = float(torch.dot(v1, g).item())

            # Monotone AGRLS requires the anchor direction to be a descent
            # direction. Adam momentum can occasionally point uphill relative to
            # the current exact gradient. In that rare case, fall back to exact
            # steepest descent for this anchor while retaining the Adam state.
            if b1 >= -1e-15:
                ng = torch.linalg.vector_norm(g)
                if float(ng) <= 1e-15:
                    break
                v1 = -g / ng
                b1 = -float(ng)
                first_step_norm = max(min(first_step_norm, adam_lr * float(ng)), 1e-12)
                anchor_direction_fallbacks += 1

            # A scalar secant curvature prior is used only until the first
            # forward observation identifies h1. The observation dominates via
            # a very small ridge term.
            if prev_refresh_theta is not None and prev_refresh_grad is not None:
                svec = theta - prev_refresh_theta
                yvec = g - prev_refresh_grad
                denom = float(torch.dot(svec, svec).item())
                if denom > 1e-18:
                    h_prior = abs(float(torch.dot(svec, yvec).item()) / denom)
                else:
                    h_prior = 0.0
            else:
                h_prior = 0.0
            h1 = h_prior

            old_refresh_theta = theta.clone()
            old_refresh_grad = g.clone()
            z_current = 0.0
            obs = []
            delta = first_step_norm
            delta_max = max(4.0 * first_step_norm, 1e-12)
            since_refresh = 0
            consecutive_rejects = 0
            cycle_first_gain = None
            first_move_pending = True
            anchor_backtracks = 0
            need_refresh = False
            rows.append({
                'algorithm': 'agrls_adam_1d_pf_mvr', 'update': accepted_updates,
                'attempt': attempts, 'event': 'gradient_refresh_adam_1d',
                'refresh_reason': refresh_reason, 'accepted': 0,
                'objective': f_current, 'forward_calls': oracle.forward_calls,
                'gradient_calls': oracle.gradient_calls,
                'forward_seconds': oracle.forward_seconds,
                'gradient_seconds': oracle.gradient_seconds,
                'trust_radius': delta, 'adam_step_norm': first_step_norm,
                'since_refresh': 0, 'curvature_h': h1,
                'predicted_improvement': float('nan'), 'actual_improvement': float('nan'),
                'rho': float('nan'), 'agreement': float('nan'),
                'retry_budget': retry_budget, 'anchor_backtracks': anchor_backtracks,
            })

        if first_move_pending:
            p = first_step_norm
            predicted = max(1e-15, -(b1 * p))
        else:
            local_g = b1 + h1 * z_current
            if abs(local_g) <= 1e-12:
                need_refresh = True
                refresh_reason = 'subspace_exhausted_1d'
                attempts += 1
                continue
            if h1 > 1e-12:
                p_unc = -local_g / h1
                p = float(np.clip(p_unc, -delta, delta))
            else:
                p = -math.copysign(delta, local_g)
            predicted = -(local_g * p + 0.5 * h1 * p * p)

            if since_refresh >= 1 and cycle_first_gain is not None and predicted > 0 and np.isfinite(predicted):
                threshold = 1.0 / (measured_r + 1.0)
                continuation_ratio = predicted / max(cycle_first_gain, 1e-15)
                if continuation_ratio < threshold:
                    need_refresh = True
                    refresh_reason = 'adam_1d_marginal_value_below_refresh_value'
                    attempts += 1
                    rows.append({
                        'algorithm': 'agrls_adam_1d_pf_mvr', 'update': accepted_updates,
                        'attempt': attempts, 'event': 'mvr_refresh_decision',
                        'refresh_reason': refresh_reason, 'accepted': 0,
                        'objective': f_current, 'forward_calls': oracle.forward_calls,
                        'gradient_calls': oracle.gradient_calls,
                        'forward_seconds': oracle.forward_seconds,
                        'gradient_seconds': oracle.gradient_seconds,
                        'trust_radius': delta, 'since_refresh': since_refresh,
                        'curvature_h': h1,
                        'predicted_improvement': predicted, 'actual_improvement': float('nan'),
                        'rho': float('nan'), 'agreement': float('nan'),
                        'continuation_value_ratio': continuation_ratio,
                        'mvr_threshold': threshold, 'retry_budget': retry_budget,
                    })
                    continue

        if (not np.isfinite(predicted)) or predicted <= 1e-12 or abs(p) <= 1e-12:
            need_refresh = True
            refresh_reason = 'subspace_exhausted_1d'
            attempts += 1
            continue

        z_trial = z_current + p
        theta_trial = anchor + float(z_trial) * v1
        f_trial = oracle.value(theta_trial)
        obs.append((float(z_trial), float(f_trial - anchor_f)))
        actual = f_current - f_trial
        rho = actual / max(predicted, 1e-15)
        agreement = min(rho, 1.0 / rho) if rho > 0 and np.isfinite(rho) else 0.0

        accepted = False
        if first_move_pending:
            # The Adam anchor is not used as evidence that an already-identified
            # quadratic is calibrated; it is the observation that identifies h.
            if actual > 0:
                accepted = True
                event = 'accept_adam_anchor'
                consecutive_rejects = 0
                delta = max(0.5 * first_step_norm, min(delta_max, first_step_norm))
            else:
                event = 'reject_adam_anchor_backtrack'
                consecutive_rejects += 1
                anchor_backtracks += 1
                first_step_norm *= 0.5
                delta = max(first_step_norm, 1e-12)
                if anchor_backtracks >= max_anchor_backtracks:
                    raise RuntimeError(
                        'Adam-anchored 1-D AGRLS failed to find a descending anchor '
                        f'after {max_anchor_backtracks} backtracks at update {accepted_updates}; '
                        'gradient refresh at unchanged theta was intentionally not repeated.'
                    )
        else:
            event = 'reject_model_miss'
            if actual > 0 and agreement >= cfg.agreement_good:
                accepted = True; event = 'accept_good'
                delta = min(delta_max, cfg.grow * delta); consecutive_rejects = 0
            elif actual > 0 and agreement >= cfg.agreement_ok:
                accepted = True; event = 'accept_ok'
                delta = max(0.25 * first_step_norm, cfg.moderate_shrink * delta); consecutive_rejects = 0
            elif actual > 0:
                accepted = True; event = 'accept_stale_model'
                delta = max(0.25 * first_step_norm, cfg.shrink * delta); consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                delta = max(0.25 * first_step_norm, cfg.shrink * delta)

        fit_center = z_trial if accepted else z_current
        h1 = refit_h_1d(
            b1, obs, h_prior, float(fit_center),
            scale=max(2.0 * delta, 2.0 * first_step_norm, 1e-12)
        )
        attempts += 1

        if accepted:
            theta = theta_trial
            f_current = f_trial
            z_current = float(z_trial)
            accepted_updates += 1
            since_refresh += 1
            if since_refresh == 1:
                cycle_first_gain = max(float(actual), 1e-15)
            first_move_pending = False
            if since_refresh >= cfg.max_without_grad:
                need_refresh = True
                refresh_reason = 'adam_1d_safety_cap'
        else:
            if first_move_pending:
                # Keep the same exact gradient and backtrack the anchor step.
                # A refresh here would query the same theta again and provides no
                # new gradient information.
                need_refresh = False
                refresh_reason = ''
            elif consecutive_rejects >= retry_budget:
                need_refresh = True
                refresh_reason = 'cost_aware_reject_budget_exhausted'

        rows.append({
            'algorithm': 'agrls_adam_1d_pf_mvr', 'update': accepted_updates,
            'attempt': attempts, 'event': event,
            'refresh_reason': refresh_reason if need_refresh else '',
            'accepted': int(accepted), 'objective': f_current,
            'trial_objective': f_trial, 'forward_calls': oracle.forward_calls,
            'gradient_calls': oracle.gradient_calls,
            'forward_seconds': oracle.forward_seconds,
            'gradient_seconds': oracle.gradient_seconds,
            'trust_radius': delta, 'adam_step_norm': first_step_norm,
            'since_refresh': since_refresh, 'curvature_h': h1,
            'predicted_improvement': predicted, 'actual_improvement': actual,
            'rho': rho, 'agreement': agreement,
            'continuation_value_ratio': (predicted / max(cycle_first_gain, 1e-15)) if cycle_first_gain else float('nan'),
            'mvr_threshold': (1.0 / (measured_r + 1.0)),
            'retry_budget': retry_budget, 'anchor_backtracks': anchor_backtracks,
        })

        if attempts > updates * 80:
            raise RuntimeError('Adam-anchored 1-D probe-free AGRLS exceeded attempt safety limit')

    copy_vector_to_params(model, theta)
    sync(device)
    elapsed = time.perf_counter() - run_t0
    train_loss, train_acc = evaluate(model, xtr, ytr, loss_fn)
    eval_loss, eval_acc = evaluate(model, xev, yev, loss_fn)
    summary = {
        'algorithm': 'agrls_adam_1d_pf_mvr', 'final_train_loss': train_loss,
        'final_train_accuracy': train_acc, 'final_eval_loss': eval_loss,
        'final_eval_accuracy': eval_acc, 'accepted_updates': accepted_updates,
        'forward_calls': oracle.forward_calls, 'gradient_calls': oracle.gradient_calls,
        'forward_seconds': oracle.forward_seconds, 'gradient_seconds': oracle.gradient_seconds,
        'oracle_seconds': oracle.forward_seconds + oracle.gradient_seconds,
        'wall_seconds': elapsed, 'measured_gradient_cost_ratio': measured_r,
        'max_moves_between_gradients': max((int(r.get('since_refresh', 0)) for r in rows), default=0),
        'dedicated_map_probe_calls': 0, 'adam_anchor_lr': adam_lr,
        'surrogate_dimension': 1, 'reject_retry_budget': retry_budget,
        'max_anchor_backtracks': max_anchor_backtracks,
        'anchor_direction_fallbacks': anchor_direction_fallbacks,
        'implementation_revision': 'v0.8.1-anchor-backtrack-fix',
    }
    return rows, summary

# ---------------------------------------------------------------------------
# Standard baselines
# ---------------------------------------------------------------------------

def run_baseline(
    name: str,
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xev: torch.Tensor,
    yev: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    updates: int,
    lr: float,
) -> Tuple[List[dict], dict]:
    model.load_state_dict(init_state)
    model.to(device)
    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif name == "momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(name)

    rows = []
    grad_seconds = 0.0
    sync(device)
    t_run = time.perf_counter()
    for step in range(1, updates + 1):
        sync(device)
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()
        sync(device)
        dt = time.perf_counter() - t0
        grad_seconds += dt
        rows.append({
            "algorithm": name,
            "update": step,
            "objective_before_step": float(loss.item()),
            "forward_calls": 0,
            "gradient_calls": step,
            "gradient_seconds": grad_seconds,
        })
    sync(device)
    elapsed = time.perf_counter() - t_run
    train_loss, train_acc = evaluate(model, xtr, ytr, loss_fn)
    eval_loss, eval_acc = evaluate(model, xev, yev, loss_fn)
    summary = {
        "algorithm": name,
        "learning_rate": lr,
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "final_eval_loss": eval_loss,
        "final_eval_accuracy": eval_acc,
        "accepted_updates": updates,
        "forward_calls": 0,
        "gradient_calls": updates,
        "forward_seconds": 0.0,
        "gradient_seconds": grad_seconds,
        "oracle_seconds": grad_seconds,
        "wall_seconds": elapsed,
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Plot / main
# ---------------------------------------------------------------------------

def plot_loss_vs_time(traces: Dict[str, List[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, rows in traces.items():
        xs, ys = [], []
        for r in rows:
            if name.startswith("agrls"):
                x = float(r.get("forward_seconds", 0.0)) + float(r.get("gradient_seconds", 0.0))
                y = float(r.get("objective", np.nan))
            else:
                x = float(r.get("gradient_seconds", 0.0))
                y = float(r.get("objective_before_step", np.nan))
            if np.isfinite(y):
                xs.append(x)
                ys.append(max(y, 1e-8))
        if xs:
            ax.semilogy(xs, ys, label=name)
    ax.set_xlabel("Measured training oracle time (s)")
    ax.set_ylabel("Training cross-entropy")
    ax.set_title("Stage 1 full-batch neural-network diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_loss_vs_gradients(traces: Dict[str, List[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, rows in traces.items():
        xs, ys = [], []
        for r in rows:
            y = float(r.get("objective", r.get("objective_before_step", np.nan)))
            if np.isfinite(y):
                xs.append(float(r.get("gradient_calls", 0)))
                ys.append(max(y, 1e-8))
        if xs:
            ax.semilogy(xs, ys, label=name)
    ax.set_xlabel("Exact-gradient / backpropagation queries")
    ax.set_ylabel("Training cross-entropy")
    ax.set_title("Stage 1 full-batch neural-network diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "fake"], default="mnist")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-subset", type=int, default=20000)
    p.add_argument("--eval-subset", type=int, default=5000)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="stage1_mnist_fullbatch_seed0")
    p.add_argument("--sgd-lr", type=float, default=0.5)
    p.add_argument("--momentum-lr", type=float, default=0.1)
    p.add_argument("--adam-lr", type=float, default=0.01)
    p.add_argument(
        "--algorithms", nargs="+",
        default=["momentum", "adam", "agrls_adam_pf_mvr", "agrls_adam_1d_pf_mvr"],
        choices=["sgd", "momentum", "adam", "agrls_fixed1", "agrls_mvr", "agrls_pf_mvr", "agrls_adam_pf_mvr", "agrls_adam_1d_pf_mvr"],
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA requested but unavailable; using CPU.")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    xtr, ytr, xev, yev = load_fullbatch_data(
        args.dataset, Path(args.data_dir), args.train_subset, args.eval_subset, args.seed, device
    )
    loss_fn = nn.CrossEntropyLoss()
    model = TinyMLP().to(device)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    theta0 = params_to_vector(model).detach().clone()

    calib_oracle = NeuralOracle(model, xtr, ytr, loss_fn, device)
    calib = calibrate_query_costs(calib_oracle, theta0)
    measured_r = float(calib["measured_gradient_cost_ratio"])
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Measured gradient-query / forward-query cost ratio: {measured_r:.3f}")

    traces: Dict[str, List[dict]] = {}
    summaries: List[dict] = []

    for alg in args.algorithms:
        print(f"Running {alg} ...")
        if alg == "sgd":
            rows, s = run_baseline(alg, model, init_state, xtr, ytr, xev, yev, loss_fn, device, args.updates, args.sgd_lr)
        elif alg == "momentum":
            rows, s = run_baseline(alg, model, init_state, xtr, ytr, xev, yev, loss_fn, device, args.updates, args.momentum_lr)
        elif alg == "adam":
            rows, s = run_baseline(alg, model, init_state, xtr, ytr, xev, yev, loss_fn, device, args.updates, args.adam_lr)
        elif alg in ("agrls_fixed1", "agrls_mvr"):
            policy = "fixed1" if alg == "agrls_fixed1" else "mvr"
            rows, s = run_agrls(
                model, init_state, xtr, ytr, xev, yev, loss_fn, device,
                args.seed, args.updates, policy, measured_r, AGRLSNNConfig()
            )
        elif alg == "agrls_pf_mvr":
            rows, s = run_agrls_probe_free(
                model, init_state, xtr, ytr, xev, yev, loss_fn, device,
                args.seed, args.updates, measured_r, AGRLSNNConfig(max_without_grad=6)
            )
        elif alg == "agrls_adam_pf_mvr":
            rows, s = run_agrls_adam_probe_free(
                model, init_state, xtr, ytr, xev, yev, loss_fn, device,
                args.seed, args.updates, measured_r, AGRLSNNConfig(max_without_grad=6),
                adam_lr=args.adam_lr
            )
        elif alg == "agrls_adam_1d_pf_mvr":
            rows, s = run_agrls_adam_probe_free_1d(
                model, init_state, xtr, ytr, xev, yev, loss_fn, device,
                args.seed, args.updates, measured_r, AGRLSNNConfig(max_without_grad=6),
                adam_lr=args.adam_lr
            )
        else:
            raise ValueError(alg)
        s.update({
            "seed": args.seed,
            "dataset": args.dataset,
            "train_subset": int(xtr.shape[0]),
            "eval_subset": int(xev.shape[0]),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "calib_forward_median_s": calib["forward_query_median_s"],
            "calib_gradient_median_s": calib["gradient_query_median_s"],
            "measured_gradient_cost_ratio": measured_r,
        })
        traces[alg] = rows
        summaries.append(s)
        write_csv(outdir / f"trace_{alg}.csv", rows)

    write_csv(outdir / "summary.csv", summaries)
    plot_loss_vs_time(traces, outdir / "train_loss_vs_measured_oracle_time.png")
    plot_loss_vs_gradients(traces, outdir / "train_loss_vs_gradient_queries.png")

    report = []
    report.append("AGRLS Stage 1D v0.8.1 — anchor-backtrack bug-fix diagnostic\n")
    report.append(f"dataset={args.dataset}; train_n={xtr.shape[0]}; eval_n={xev.shape[0]}; seed={args.seed}; updates={args.updates}\n")
    report.append(f"device={device}; gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}\n")
    report.append(f"measured gradient/forward query cost ratio r={measured_r:.6f}\n")
    report.append(f"median forward query={calib['forward_query_median_s']:.6f}s; median gradient query={calib['gradient_query_median_s']:.6f}s\n\n")
    for s in summaries:
        report.append(
            f"{s['algorithm']:14s} train_loss={s['final_train_loss']:.6f} "
            f"train_acc={s['final_train_accuracy']:.4f} eval_loss={s['final_eval_loss']:.6f} "
            f"eval_acc={s['final_eval_accuracy']:.4f} grad={s['gradient_calls']} "
            f"fwd={s['forward_calls']} oracle_s={s['oracle_seconds']:.4f} wall_s={s['wall_seconds']:.4f}\n"
        )
    (outdir / "REPORT.txt").write_text("".join(report), encoding="utf-8")
    print("\n" + "".join(report))
    print(f"Results: {outdir.resolve()}")


if __name__ == "__main__":
    main()
