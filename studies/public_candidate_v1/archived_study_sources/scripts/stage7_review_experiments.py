"""Post-confirmation LGPB-MC reviewer ablations (NO modification to Stage 6 code).

Install beside repo src/ and run through run_stage7_review.cmd. All publication-scale
experiments must be run on the researcher's local GPU. See REVIEW_STAGE7_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # Local flat working tree; src/ remains supported.

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

import agrls_stage6a_cifar10_augmented_schedule as frozen
import agrls_stage5a1_cifar10_fggr_microprobe as netutils

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
MATCHED_Q = 0.35990173847316703  # Predeclared from published Stage 6B 9-seed mean, NOT re-fit here
IMPLEMENTATION_VERSION = "stage7-local-v1.1"
MATCHED_EPOCH_Q = (
    0.4682539683, 0.4024943311, 0.3832199546, 0.3764172336, 0.3690476190,
    0.3713151927, 0.3639455782, 0.3599773243, 0.3548752834, 0.3633786848,
    0.3475056689, 0.3622448980, 0.3486394558, 0.3520408163, 0.3452380952,
    0.3537414966, 0.3514739229, 0.3480725624, 0.3452380952, 0.3429705215,
    0.3514739229, 0.3429705215, 0.3486394558, 0.3435374150, 0.3514739229,
    0.3503401361, 0.3543083900, 0.3497732426, 0.3480725624, 0.3463718821,
)  # precomputed Stage6B mean q for each epoch; frozen before Stage7
VARIANTS = {
    "adam", "lgpb", "random_matched", "periodic_matched",
    "freeze_prefix", "replay_prefix", "rho_090", "rho_099",
    "no_forced_full", "prefix_stem", "prefix_stem_layer12",
}


class TinyCifar(nn.Module):
    """Smoke-test-only tiny model matching ResNet parameter-name partitions."""
    def __init__(self, nclass: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, 8)
        self.layer1 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(8, nclass)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: torch.Tensor, dataset: str) -> torch.Tensor:
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if dataset == "cifar100" else (
        netutils.CIFAR10_MEAN, netutils.CIFAR10_STD)
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def load_data(args, device: torch.device):
    if args.dataset == "fake":
        g = torch.Generator(device="cpu").manual_seed(args.seed + 53001)
        xtr = torch.rand(args.train_subset, 3, 32, 32, generator=g).to(device)
        ytr = torch.randint(0, args.num_classes, (args.train_subset,), generator=g).to(device)
        xev = torch.rand(args.eval_subset, 3, 32, 32, generator=g).to(device)
        yev = torch.randint(0, args.num_classes, (args.eval_subset,), generator=g).to(device)
        return xtr, ytr, normalize(xev, "cifar10"), yev
    ds_cls = datasets.CIFAR100 if args.dataset == "cifar100" else datasets.CIFAR10
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    train = ds_cls(root=str(root), train=True, download=True)
    test = ds_cls(root=str(root), train=False, download=True)
    xtr = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    xev = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    ytr = torch.tensor(train.targets, dtype=torch.long)
    yev = torch.tensor(test.targets, dtype=torch.long)
    if args.train_subset > len(xtr) or args.eval_subset > len(xev):
        raise ValueError("requested subset larger than available dataset")
    if args.train_subset < len(xtr):
        g = torch.Generator(device="cpu").manual_seed(args.seed + 51001)
        chosen = torch.randperm(len(xtr), generator=g)[: args.train_subset]
        xtr, ytr = xtr.index_select(0, chosen), ytr.index_select(0, chosen)
    if args.eval_subset < len(xev):
        g = torch.Generator(device="cpu").manual_seed(args.seed + 52001)
        chosen = torch.randperm(len(xev), generator=g)[: args.eval_subset]
        xev, yev = xev.index_select(0, chosen), yev.index_select(0, chosen)
    return xtr.to(device), ytr.to(device), normalize(xev.to(device), args.dataset), yev.to(device)


def augment(x: torch.Tensor, generator: torch.Generator, dataset: str) -> torch.Tensor:
    if dataset in ("cifar10", "fake"):
        # Bit-for-bit same operation sequence / RNG calls as frozen Stage 6 harness.
        return frozen.augment_and_normalize(x, generator)
    n = int(x.shape[0])
    oy = torch.randint(0, 9, (n,), generator=generator).to(x.device)
    ox = torch.randint(0, 9, (n,), generator=generator).to(x.device)
    flip = (torch.rand(n, generator=generator) < 0.5).to(x.device)
    padded = F.pad(x, (4, 4, 4, 4), mode="constant", value=0.0)
    windows = padded.unfold(2, 32, 1).unfold(3, 32, 1)
    cropped = windows[torch.arange(n, device=x.device), :, oy, ox].contiguous()
    cropped = torch.where(flip[:, None, None, None], cropped.flip(3), cropped)
    return normalize(cropped, dataset)


def make_model(args, device):
    if args.tiny_model:
        model = TinyCifar(args.num_classes)
    else:
        model = netutils.make_cifar_resnet18_gn()
        if args.num_classes != 10:
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    return model.to(device)


def partition(model: nn.Module, variant: str):
    if variant == "prefix_stem":
        suffix_roots = ("layer1.", "layer2.", "layer3.", "layer4.", "fc.")
    elif variant == "prefix_stem_layer12":
        suffix_roots = ("layer3.", "layer4.", "fc.")
    else:
        suffix_roots = frozen.SUFFIX_PREFIXES
    suffix, prefix = [], []
    for name, p in model.named_parameters():
        if name.startswith(suffix_roots):
            suffix.append(p)
        else:
            prefix.append(p)
    if not suffix or not prefix:
        raise RuntimeError("empty suffix or prefix; inspect model parameter names")
    return prefix, suffix


def matched_partial_indices(n: int, q: float, seed: int, randomize: bool):
    """Exactly round(N*q) nonadjacent partial positions, independent of losses.

    Random mode samples uniformly from feasible no-consecutive configurations.
    Periodic mode uses a near-even spacing with the same global count.
    """
    k = int(round(n * q))
    if k < 0 or k > n // 2:
        raise ValueError(f"target {q} infeasible with mandatory first-full + nonconsecutive partials")
    if randomize:
        rng = random.Random(seed + 95113)
        positions = sorted(rng.sample(range(n - k), k))
        return set(j + i + 1 for i, j in enumerate(positions))
    # Direct periodic placement, exactly k positions; first partial always at update >=2.
    # Since q<1/2, target rounding provides nonconsecutive positions.
    indices = {int((j + 0.5) * (n - 1) / k) + 1 for j in range(k)} if k else set()
    if len(indices) != k or any(x + 1 in indices for x in indices) or 0 in indices:
        # Robust Bresenham fallback with nonconsecutive steps and correct count.
        indices = set()
        for t in range(1, n):
            candidate = ((t + 1) * k) // n > (t * k) // n
            if candidate and t - 1 not in indices:
                indices.add(t)
        if len(indices) != k:
            raise RuntimeError("periodic partial schedule count mismatch")
    return indices


def epoch_matched_partial_indices(epochs: int, batches_per_epoch: int, seed: int, randomize: bool):
    """Loss-blind skip schedule matched to the ORIGINAL nine-seed mean q(epoch).

    Includes precisely round(q_epoch*n) partials in each epoch. A mandatory full
    on the first mini-batch of each epoch ensures no cross-epoch adjacency.
    This uses archived Stage6B history, never the outcomes of new Stage7 runs.
    """
    if epochs > len(MATCHED_EPOCH_Q):
        raise ValueError("only original 30-epoch archived q available")
    indices = set()
    for epoch in range(epochs):
        local = matched_partial_indices(batches_per_epoch, MATCHED_EPOCH_Q[epoch],
                                        seed + 101 * epoch, randomize)
        indices.update(epoch * batches_per_epoch + j for j in local)
    if any(v + 1 in indices for v in indices):
        raise AssertionError("epochwise matched schedule has consecutive partials")
    return indices


def choose_partial(variant: str, t: int, ema: float | None,
                   previous_partial: bool, loss_value: float,
                   planned: set[int]) -> bool:
    if variant == "adam":
        return False
    if variant in ("random_matched", "periodic_matched"):
        return t in planned
    return (ema is not None and (variant == "no_forced_full" or not previous_partial)
            and loss_value <= ema)


def adam_optimizer(args, model: nn.Module):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)


@torch.no_grad()
def add_replayed_delta(prefix: List[nn.Parameter], last_delta: List[torch.Tensor]) -> None:
    for p, d in zip(prefix, last_delta):
        p.add_(d)


def evaluate(model, x, y, batch_size):
    # Evaluation is excluded from timed training wall exactly as Stage 6.
    loss_fn = nn.CrossEntropyLoss()
    return netutils.evaluate_batched(model, x, y, loss_fn, batch_size)


def run_one(args, variant: str, model: nn.Module, init_state,
            xtr, ytr, xev, yev, device, *, warmup: bool = False):
    model.load_state_dict(init_state)
    model.train()
    prefix, suffix = partition(model, variant)
    optimizer = adam_optimizer(args, model)
    rho = 0.90 if variant == "rho_090" else (0.99 if variant == "rho_099" else args.rho)
    n = int(len(xtr))
    nepoch = (min(args.epochs, 1) if warmup else args.epochs)
    batches_per_epoch = math.ceil(n / args.batch_size)
    total_steps = args.schedule_epochs * batches_per_epoch
    selected_partials = set()
    if variant in ("random_matched", "periodic_matched"):
        if args.match_policy == "epoch":
            selected_partials = epoch_matched_partial_indices(
                nepoch, batches_per_epoch, args.seed,
                randomize=(variant == "random_matched"))
        else:
            selected_partials = matched_partial_indices(
                nepoch * batches_per_epoch, args.matched_q, args.seed,
                randomize=(variant == "random_matched"))
    epoch_rows = []
    step_rows = []
    ema = None
    prev_partial = False
    partial_streak = 0
    max_partial_streak = 0
    partial_total = full_total = 0
    last_delta = None
    clocks = {"augmentation": 0.0, "forward": 0.0, "backward": 0.0,
              "optimizer": 0.0, "backward_full": 0.0, "backward_partial": 0.0}
    cnt = {"forward": 0, "backward_full": 0, "backward_partial": 0, "optimizer": 0}
    batch_gen = torch.Generator(device="cpu").manual_seed(args.seed + (93001 if warmup else 91001))
    aug_gen = torch.Generator(device="cpu").manual_seed(args.seed + (94001 if warmup else 92001))
    idx = 0
    cumulative = 0.0
    update_steps_to_log = args.log_steps and not warmup
    sync(device)
    for epoch in range(1, nepoch + 1):
        epoch_loss_sum = 0.0
        epoch_count = 0
        epoch_partial = 0
        epoch_full = 0
        epoch_times = {"full": [], "partial": []}
        epoch_backward_times = {"full": [], "partial": []}
        sync(device); epoch_clock = time.perf_counter()
        for cpu_ix in frozen.epoch_batches(n, args.batch_size, batch_gen):
            raw = xtr.index_select(0, cpu_ix.to(device))
            y = ytr.index_select(0, cpu_ix.to(device))
            sync(device); t0 = time.perf_counter()
            x = augment(raw, aug_gen, args.dataset)
            sync(device); aug_seconds = time.perf_counter() - t0
            lr = frozen.set_cosine_lr(optimizer, args.lr, idx, total_steps, args.min_lr_ratio)
            optimizer.zero_grad(set_to_none=True)
            sync(device); t0 = time.perf_counter()
            loss = F.cross_entropy(model(x), y)
            sync(device); f_seconds = time.perf_counter() - t0
            lv = float(loss.detach().item())
            if not math.isfinite(lv):
                raise FloatingPointError(f"Nonfinite loss: {variant}, epoch={epoch}, step={idx + 1}")
            partial = choose_partial(variant, idx, ema, prev_partial, lv, selected_partials)
            if partial and prev_partial and variant != "no_forced_full":
                raise AssertionError("illegal consecutive partial update")
            if partial and (ema is None) and variant not in ("random_matched", "periodic_matched"):
                raise AssertionError("loss EMA uninitialized before partial")
            sync(device); t0 = time.perf_counter()
            if partial:
                gradients = torch.autograd.grad(loss, suffix)
                for p, g in zip(suffix, gradients):
                    p.grad = g
                if variant in ("freeze_prefix", "replay_prefix"):
                    for p in prefix:
                        p.grad = None  # Adam skips prefix parameter and ALL optimizer state updates.
                else:
                    for p in prefix:
                        p.grad = torch.zeros_like(p)  # THE frozen LGPB-MC moment continuation.
            else:
                loss.backward()
            sync(device); b_seconds = time.perf_counter() - t0
            sync(device); t0 = time.perf_counter()
            # Replay comparator keeps the last full Adam displacement but does NOT advance prefix moments
            # on partial steps. This includes recording a delta on full steps: true comparator overhead.
            if variant == "replay_prefix" and not partial:
                before = [p.detach().clone() for p in prefix]
            optimizer.step()
            if variant == "replay_prefix":
                if partial:
                    if last_delta is None:
                        raise AssertionError("replay without prior full update")
                    add_replayed_delta(prefix, last_delta)
                else:
                    last_delta = [(p.detach() - b) for p, b in zip(prefix, before)]
            sync(device); o_seconds = time.perf_counter() - t0
            # Gate reference updates only AFTER optimizer.step, using PRE-update loss.
            ema_before = ema
            ema = lv if ema is None else rho * ema + (1.0 - rho) * lv
            prev_partial = partial
            if partial:
                partial_total += 1
                partial_streak += 1
                epoch_partial += 1
            else:
                full_total += 1
                partial_streak = 0
                epoch_full += 1
            max_partial_streak = max(max_partial_streak, partial_streak)
            clocks["augmentation"] += aug_seconds
            clocks["forward"] += f_seconds
            clocks["backward"] += b_seconds
            clocks["optimizer"] += o_seconds
            clocks["backward_partial" if partial else "backward_full"] += b_seconds
            cnt["backward_partial" if partial else "backward_full"] += 1
            cnt["forward"] += 1
            cnt["optimizer"] += 1
            epoch_times["partial" if partial else "full"].append((aug_seconds + f_seconds + b_seconds + o_seconds))
            epoch_backward_times["partial" if partial else "full"].append(b_seconds)
            num = len(y)
            epoch_loss_sum += lv * num
            epoch_count += num
            if update_steps_to_log:
                step_rows.append({"epoch": epoch, "step": idx + 1, "partial": int(partial),
                                  "ema_before_update": "" if ema_before is None else ema_before,
                                  "loss": lv, "aug_ms": aug_seconds * 1000,
                                  "forward_ms": f_seconds * 1000, "backward_ms": b_seconds * 1000,
                                  "optimizer_ms": o_seconds * 1000, "lr": lr})
            idx += 1
        sync(device); epoch_seconds = time.perf_counter() - epoch_clock
        cumulative += epoch_seconds
        if not warmup:
            evloss, evacc = evaluate(model, xev, yev, args.eval_batch_size)
            epoch_rows.append({"variant": variant, "epoch": epoch, "updates": idx,
                               "training_loss": epoch_loss_sum / max(1, epoch_count),
                               "eval_accuracy": evacc, "eval_loss": evloss,
                               "epoch_seconds": epoch_seconds, "cumulative_training_seconds": cumulative,
                               "partial_updates": epoch_partial, "full_updates": epoch_full,
                               "partial_fraction": epoch_partial / (epoch_partial + epoch_full),
                               "mean_step_full_seconds": (sum(epoch_times["full"]) / max(1, len(epoch_times["full"]))),
                               "mean_step_partial_seconds": (sum(epoch_times["partial"]) / max(1, len(epoch_times["partial"]))),
                               "mean_backward_full_seconds": (sum(epoch_backward_times["full"]) / max(1, len(epoch_backward_times["full"]))),
                               "mean_backward_partial_seconds": (sum(epoch_backward_times["partial"]) / max(1, len(epoch_backward_times["partial"]))),
                               "learning_rate_end": lr})
            print(f"[{variant}] seed={args.seed} epoch={epoch}/{nepoch} "
                  f"eval={evacc * 100:.2f}% partial={partial_total}/{idx} "
                  f"train_s={cumulative:.1f}", flush=True)
    final_accuracy = None if warmup else epoch_rows[-1]["eval_accuracy"]
    final_loss = None if warmup else epoch_rows[-1]["eval_loss"]
    targets = (0.40,0.50,0.60) if args.dataset == "cifar100" else (0.60,0.70,0.80)
    target_data = {}
    if not warmup:
        for threshold in targets:
            first = next((r for r in epoch_rows if r["eval_accuracy"] >= threshold), None)
            label = f"{round(100*threshold)}pct"
            target_data[f"time_to_{label}_seconds"] = first["cumulative_training_seconds"] if first else None
            target_data[f"epoch_to_{label}"] = first["epoch"] if first else None
    summary = {
        "variant": variant, "dataset": args.dataset,
        "seed": args.seed, "epochs": nepoch, "updates": idx,
        "optimizer": args.optimizer, "lr": args.lr,
        "rho": rho, "matched_q_target": args.matched_q,
        "train_subset": len(xtr), "eval_subset": len(xev), "batch_size": args.batch_size,
        "partial_updates": partial_total, "full_updates": full_total,
        "partial_fraction": partial_total / max(1, idx),
        "max_consecutive_partials": max_partial_streak,
        "partial_schedule": ("archived_epoch_q_nonadjacent" if args.match_policy == "epoch" else "fixed_global_q_nonadjacent") if variant in ("random_matched","periodic_matched") else "loss_ema_or_full",
        "prefix_parameter_count": sum(p.numel() for p in prefix),
        "suffix_parameter_count": sum(p.numel() for p in suffix),
        "prefix_parameter_fraction": sum(p.numel() for p in prefix) / sum(p.numel() for p in model.parameters()),
        "final_eval_accuracy": final_accuracy, "final_eval_loss": final_loss,
        "training_wall_seconds": cumulative,
        "augmentation_seconds": clocks["augmentation"], "forward_seconds": clocks["forward"],
        "backward_seconds": clocks["backward"], "backward_full_seconds": clocks["backward_full"],
        "backward_partial_seconds": clocks["backward_partial"], "optimizer_seconds": clocks["optimizer"],
        "avg_backward_full_ms": (1000 * clocks["backward_full"] / max(1, cnt["backward_full"])),
        "avg_backward_partial_ms": (1000 * clocks["backward_partial"] / max(1, cnt["backward_partial"])),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "warmup_steps_excluded": args.warmup_steps,
    }
    summary.update(target_data)
    summary["timing_unattributed_seconds"] = cumulative - sum(
        clocks[key] for key in ("augmentation", "forward", "backward", "optimizer"))
    summary["implementation_version"] = IMPLEMENTATION_VERSION
    summary["status"] = "complete"
    return epoch_rows, step_rows, summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["core", "sensitivity", "transfer", "smoke"], default="core")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "fake"], default="cifar10")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-subset", type=int, default=50000)
    p.add_argument("--eval-subset", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--schedule-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--device", default="cuda")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--matched-q", type=float, default=MATCHED_Q)
    p.add_argument("--match-policy", choices=["epoch", "global"], default="epoch",
                   help="Default: match the frozen Stage6B q for EACH epoch, not only the grand mean")
    p.add_argument("--warmup-steps", type=int, default=196)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--log-steps", action="store_true", help="Output per-step records; substantial CSV volume")
    p.add_argument("--tiny-model", action="store_true", help="SMOKE TEST ONLY: not ResNet")
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--out", default="results/stage7_review")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rotate-order", action="store_true", help="Rotate algorithm execution order across prespecified seeds")
    p.add_argument("--rotation-start", type=int, default=51)
    args = p.parse_args()
    for name in ("train_subset", "eval_subset", "batch_size", "epochs", "schedule_epochs", "eval_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup_steps < 0 or not 0 <= args.rho < 1 or args.lr <= 0:
        raise ValueError("Invalid warmup, rho, or learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable; CPU is for smoke tests only")
    args.num_classes = 100 if args.dataset == "cifar100" else 10
    if args.variants is None:
        args.variants = {
            "core": ["adam", "lgpb", "random_matched", "periodic_matched", "freeze_prefix", "replay_prefix"],
            "sensitivity": ["lgpb", "rho_090", "rho_099", "no_forced_full", "prefix_stem", "prefix_stem_layer12"],
            "transfer": ["adam", "lgpb"],
            "smoke": ["adam", "lgpb", "random_matched", "periodic_matched", "freeze_prefix", "replay_prefix", "prefix_stem_layer12"],
        }[args.suite]
    if set(args.variants) - VARIANTS:
        raise ValueError(f"Unknown variant(s): {set(args.variants) - VARIANTS}")
    if args.rotate_order:
        offset = (args.seed - args.rotation_start) % len(args.variants)
        args.variants = args.variants[offset:] + args.variants[:offset]
    if args.schedule_epochs < args.epochs:
        raise ValueError("schedule-epochs must be >= epochs")
    if not 0.0 < args.matched_q < 0.5:
        raise ValueError("matched_q must be in (0,0.5)")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA required but not available. Use --device cpu for smoke tests")
    seed_everything(args.seed)
    xtr, ytr, xev, yev = load_data(args, device)
    model = make_model(args, device)
    init_state = copy.deepcopy(model.state_dict())
    target = Path(args.out) / f"seed_{args.seed:03d}"
    target.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args), "device_resolved": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "implementation_version": IMPLEMENTATION_VERSION,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), Path(frozen.__file__).resolve(), Path(netutils.__file__).resolve())},
        "torch_version": torch.__version__, "python_version": platform.python_version(),
    }
    fingerprint_config = {k: v for k, v in config.items() if k not in ("force", "out")}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_config, sort_keys=True).encode()).hexdigest()
    config["config_fingerprint"] = fingerprint
    config_path = target / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("Existing output has different settings/code/environment. Use a NEW --out directory.")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    for variant in args.variants:
        folder = target / variant
        if (folder / "summary.json").exists() and not args.force:
            existing = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != fingerprint:
                raise ValueError(f"Completion record does not match configuration: {folder}")
            print(f"[SKIP] {variant} seed={args.seed} already complete")
            continue
        folder.mkdir(exist_ok=True)
        # Remove completion marker before a forced rerun; interrupted runs stay incomplete.
        (folder / "summary.json").unlink(missing_ok=True)
        if args.warmup_steps > 0:
            # EXACT freeze/replay/partial warmup path, but separate data RNG stream.
            # For method-specific warmup, process N warm steps by repeating fake/supplied
            # training mini-batches inside one epoch; reset weights+optimizer before timed run.
            run_warmup_steps(args, variant, model, init_state, xtr, ytr, device)
        model.load_state_dict(init_state)
        rows, steps, summary = run_one(args, variant, model, init_state, xtr, ytr, xev, yev, device)
        summary["run_order_position"] = args.variants.index(variant)
        summary["config_fingerprint"] = fingerprint
        write_csv(folder / "epoch_trace.csv", rows)
        if args.log_steps:
            write_csv(folder / "step_trace.csv", steps)
        pending = folder / "summary.json.tmp"
        pending.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(folder / "summary.json")
        print(f"COMPLETE {variant} seed={args.seed}: acc={100*summary['final_eval_accuracy']:.2f}% "
              f"wall={summary['training_wall_seconds']:.2f}s partial={100*summary['partial_fraction']:.2f}%")
    print("Output:", target)


def run_warmup_steps(args, variant, model, init_state, xtr, ytr, device):
    """Warm the candidate's own computational paths with a separate RNG stream.

    Updates DO NOT count toward training. The complete initial state is restored later.
    This is a warmup only; no metrics/decisions from warmup are used as evidence.
    """
    model.load_state_dict(init_state)
    model.train()
    prefix, suffix = partition(model, variant)
    optimizer = adam_optimizer(args, model)
    ngen = torch.Generator(device="cpu").manual_seed(args.seed + 93001)
    agen = torch.Generator(device="cpu").manual_seed(args.seed + 94001)
    ema = None
    prev = False
    last_delta = None
    rho = .9 if variant == "rho_090" else (.99 if variant == "rho_099" else args.rho)
    indices = []
    while len(indices) < args.warmup_steps:
        for x in frozen.epoch_batches(len(xtr), args.batch_size, ngen):
            indices.append(x)
            if len(indices) == args.warmup_steps:
                break
    # Warm-up schedule is separate and does NOT reuse the held-out epoch decisions.
    plan = matched_partial_indices(args.warmup_steps, args.matched_q, args.seed, variant == "random_matched") if variant in ("random_matched", "periodic_matched") else set()
    for t, ix in enumerate(indices):
        xb = augment(xtr.index_select(0, ix.to(device)), agen, args.dataset)
        yb = ytr.index_select(0, ix.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        lv = float(loss.detach().item())
        partial = choose_partial(variant, t, ema, prev, lv, plan)
        if partial:
            gs = torch.autograd.grad(loss, suffix)
            for p, g in zip(suffix, gs): p.grad = g
            for p in prefix: p.grad = None if variant in ("freeze_prefix", "replay_prefix") else torch.zeros_like(p)
        else:
            loss.backward()
        if variant == "replay_prefix" and not partial:
            before = [p.detach().clone() for p in prefix]
        optimizer.step()
        if variant == "replay_prefix":
            if partial:
                add_replayed_delta(prefix, last_delta)
            else:
                last_delta = [(p.detach() - b) for p, b in zip(prefix, before)]
        ema = lv if ema is None else rho * ema + (1-rho)*lv
        prev = partial
    sync(device)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(init_state)
    del optimizer
    sync(device)


if __name__ == "__main__":
    main()
