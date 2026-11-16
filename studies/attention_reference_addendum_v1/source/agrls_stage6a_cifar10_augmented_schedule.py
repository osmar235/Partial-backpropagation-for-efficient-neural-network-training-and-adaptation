from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

import agrls_stage1_mnist_fullbatch_v081 as v081
import agrls_stage4a_minibatch_locked_v012 as v012
import agrls_stage4g_forward_gated_refresh as v4g
import agrls_stage5a1_cifar10_fggr_microprobe as s5a1
import agrls_stage5d1_cifar10_optimizer_lr_screen as s5d1


# Stage 6A: publication-protocol transfer with the Stage 5 controller frozen.
# Frozen controller: loss <= pre-update EMA, EMA decay 0.95, at most one
# consecutive partial update, prefix=stem+layer1, and explicit zero gradients
# on the prefix during partial Adam steps (moment continuation).

SUFFIX_PREFIXES = ("layer2.", "layer3.", "layer4.", "fc.")
TARGET_ACCURACIES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def _subset_indices(n: int, k: int, seed: int) -> torch.Tensor:
    if k <= 0 or k > n:
        raise ValueError(f"subset size {k} must be in [1, {n}]")
    if k == n:
        return torch.arange(n)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=generator)[:k]


def normalize_cifar(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(s5a1.CIFAR10_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(s5a1.CIFAR10_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_cifar10_raw(
    root: Path,
    train_subset: int,
    eval_subset: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root.mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(root=str(root), train=True, download=True)
    test = datasets.CIFAR10(root=str(root), train=False, download=True)
    x_train = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    y_train = torch.tensor(train.targets, dtype=torch.long)
    x_eval = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    y_eval = torch.tensor(test.targets, dtype=torch.long)
    train_idx = _subset_indices(len(x_train), train_subset, seed + 51001)
    eval_idx = _subset_indices(len(x_eval), eval_subset, seed + 52001)
    x_train = x_train.index_select(0, train_idx).to(device)
    y_train = y_train.index_select(0, train_idx).to(device)
    x_eval = normalize_cifar(x_eval.index_select(0, eval_idx).to(device))
    y_eval = y_eval.index_select(0, eval_idx).to(device)
    return x_train, y_train, x_eval, y_eval


def load_fake_raw(
    train_subset: int,
    eval_subset: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 53001)
    x_train = torch.rand(train_subset, 3, 32, 32, generator=generator).to(device)
    y_train = torch.randint(0, 10, (train_subset,), generator=generator).to(device)
    x_eval = torch.rand(eval_subset, 3, 32, 32, generator=generator).to(device)
    y_eval = torch.randint(0, 10, (eval_subset,), generator=generator).to(device)
    return x_train, y_train, normalize_cifar(x_eval), y_eval


def augment_and_normalize(
    x: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Random 32x32 crop after four-pixel zero padding, then horizontal flip."""
    batch = int(x.shape[0])
    offset_y = torch.randint(0, 9, (batch,), generator=generator).to(x.device)
    offset_x = torch.randint(0, 9, (batch,), generator=generator).to(x.device)
    flip = (torch.rand(batch, generator=generator) < 0.5).to(x.device)
    padded = F.pad(x, (4, 4, 4, 4), mode="constant", value=0.0)
    windows = padded.unfold(2, 32, 1).unfold(3, 32, 1)
    rows = torch.arange(batch, device=x.device)
    cropped = windows[rows, :, offset_y, offset_x].contiguous()
    cropped = torch.where(flip[:, None, None, None], cropped.flip(3), cropped)
    return normalize_cifar(cropped)


def epoch_batches(
    n: int,
    batch_size: int,
    generator: torch.Generator,
) -> Iterable[torch.Tensor]:
    permutation = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield permutation[start : start + batch_size]


def set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    progress = step / max(total_steps - 1, 1)
    multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def warm_algorithm(
    algorithm: str,
    base_lr: float,
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    x_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    batch_size: int,
    seed: int,
    steps: int,
    ema_decay: float,
) -> float:
    """Warm algorithm-specific kernels and allocator state outside the timer."""
    if steps <= 0:
        return 0.0
    model.load_state_dict(init_state)
    model.train()
    suffix_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(SUFFIX_PREFIXES)
    ]
    prefix_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith(SUFFIX_PREFIXES)
    ]
    optimizer_name = "adan" if algorithm == "adan" else "adam"
    optimizer = s5d1.make_optimizer(optimizer_name, model, base_lr)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 93001)
    augmentation_generator = torch.Generator(device="cpu").manual_seed(seed + 94001)
    stream = iter(())
    loss_ema = None
    previous_partial = False
    v081.sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        try:
            idx_cpu = next(stream)
        except StopIteration:
            stream = iter(epoch_batches(int(x_train_raw.shape[0]), batch_size, batch_generator))
            idx_cpu = next(stream)
        idx = idx_cpu.to(device)
        x_batch = augment_and_normalize(x_train_raw.index_select(0, idx), augmentation_generator)
        y_batch = y_train.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_batch), y_batch)
        loss_value = float(loss.detach().item())
        if algorithm in ("adam", "adan"):
            use_partial = False
        else:
            eligible = loss_ema is not None and not previous_partial
            use_partial = bool(eligible and loss_value <= loss_ema)
        if use_partial:
            gradients = torch.autograd.grad(loss, suffix_parameters)
            for parameter, gradient in zip(suffix_parameters, gradients):
                parameter.grad = gradient
            for parameter in prefix_parameters:
                parameter.grad = torch.zeros_like(parameter)
        else:
            loss.backward()
        optimizer.step()
        loss_ema = loss_value if loss_ema is None else (
            ema_decay * loss_ema + (1.0 - ema_decay) * loss_value
        )
        previous_partial = use_partial
    v081.sync(device)
    seconds = time.perf_counter() - start
    del optimizer
    model.zero_grad(set_to_none=True)
    model.load_state_dict(init_state)
    v081.sync(device)
    return seconds


def run_one(
    algorithm: str,
    base_lr: float,
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    x_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    schedule_epochs: int,
    batch_size: int,
    seed: int,
    eval_batch_size: int,
    min_lr_ratio: float,
    ema_decay: float,
    targets: Sequence[float],
    warmup_steps: int,
) -> Tuple[List[dict], dict]:
    model.load_state_dict(init_state)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    suffix_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(SUFFIX_PREFIXES)
    ]
    prefix_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith(SUFFIX_PREFIXES)
    ]
    optimizer_name = "adan" if algorithm == "adan" else "adam"
    warmup_seconds = warm_algorithm(
        algorithm, base_lr, model, init_state, x_train_raw, y_train, loss_fn,
        device, batch_size, seed, warmup_steps, ema_decay,
    )
    optimizer = s5d1.make_optimizer(optimizer_name, model, base_lr)
    batches_per_epoch = math.ceil(int(x_train_raw.shape[0]) / batch_size)
    total_steps = schedule_epochs * batches_per_epoch
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 91001)
    augmentation_generator = torch.Generator(device="cpu").manual_seed(seed + 92001)
    loss_ema = None
    previous_partial = False
    full_updates = partial_updates = 0
    forward_seconds = backward_seconds = optimizer_seconds = augmentation_seconds = 0.0
    cumulative_training_seconds = 0.0
    global_step = 0
    epoch_rows: List[dict] = []
    first_target_seconds = {float(target): None for target in targets}
    first_target_epoch = {float(target): None for target in targets}

    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_full_before = full_updates
        epoch_partial_before = partial_updates
        v081.sync(device)
        epoch_start = time.perf_counter()
        for idx_cpu in epoch_batches(int(x_train_raw.shape[0]), batch_size, batch_generator):
            idx = idx_cpu.to(device)
            raw_batch = x_train_raw.index_select(0, idx)
            y_batch = y_train.index_select(0, idx)
            v081.sync(device); start = time.perf_counter()
            x_batch = augment_and_normalize(raw_batch, augmentation_generator)
            v081.sync(device); augmentation_seconds += time.perf_counter() - start
            lr = set_cosine_lr(optimizer, base_lr, global_step, total_steps, min_lr_ratio)
            optimizer.zero_grad(set_to_none=True)
            v081.sync(device); start = time.perf_counter()
            loss = loss_fn(model(x_batch), y_batch)
            v081.sync(device); forward_seconds += time.perf_counter() - start
            loss_value = float(loss.detach().item())

            if algorithm in ("adam", "adan"):
                use_partial = False
            elif algorithm == "loss_gated_moment_continuation_adam":
                eligible = loss_ema is not None and not previous_partial
                use_partial = bool(eligible and loss_value <= loss_ema)
            else:
                raise ValueError(f"unknown algorithm: {algorithm}")

            v081.sync(device); start = time.perf_counter()
            if use_partial:
                gradients = torch.autograd.grad(loss, suffix_parameters)
                for parameter, gradient in zip(suffix_parameters, gradients):
                    parameter.grad = gradient
                for parameter in prefix_parameters:
                    parameter.grad = torch.zeros_like(parameter)
                partial_updates += 1
            else:
                loss.backward()
                full_updates += 1
            v081.sync(device); backward_seconds += time.perf_counter() - start
            v081.sync(device); start = time.perf_counter()
            optimizer.step()
            v081.sync(device); optimizer_seconds += time.perf_counter() - start

            loss_ema = loss_value if loss_ema is None else (
                ema_decay * loss_ema + (1.0 - ema_decay) * loss_value
            )
            previous_partial = use_partial
            count = int(y_batch.shape[0])
            epoch_loss_sum += loss_value * count
            epoch_examples += count
            global_step += 1

        v081.sync(device)
        epoch_training_seconds = time.perf_counter() - epoch_start
        cumulative_training_seconds += epoch_training_seconds
        eval_start = time.perf_counter()
        eval_loss, eval_accuracy = s5a1.evaluate_batched(
            model, x_eval, y_eval, loss_fn, eval_batch_size
        )
        v081.sync(device)
        eval_seconds = time.perf_counter() - eval_start
        for target in targets:
            target = float(target)
            if first_target_seconds[target] is None and eval_accuracy >= target:
                first_target_seconds[target] = cumulative_training_seconds
                first_target_epoch[target] = epoch
        row = {
            "algorithm": algorithm,
            "epoch": epoch,
            "updates": global_step,
            "train_augmented_loss": epoch_loss_sum / max(epoch_examples, 1),
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "epoch_training_seconds": epoch_training_seconds,
            "cumulative_training_seconds": cumulative_training_seconds,
            "eval_seconds": eval_seconds,
            "epoch_full_updates": full_updates - epoch_full_before,
            "epoch_partial_updates": partial_updates - epoch_partial_before,
            "cumulative_full_updates": full_updates,
            "cumulative_partial_updates": partial_updates,
            "partial_update_fraction": partial_updates / max(global_step, 1),
            "learning_rate_end": lr,
        }
        epoch_rows.append(row)
        print(
            f"{algorithm}: epoch {epoch}/{epochs}; "
            f"eval={100 * eval_accuracy:.2f}%; partial={partial_updates}/{global_step}; "
            f"train_s={cumulative_training_seconds:.1f}",
            flush=True,
        )

    clean_train = normalize_cifar(x_train_raw)
    train_loss, train_accuracy = s5a1.evaluate_batched(
        model, clean_train, y_train, loss_fn, eval_batch_size
    )
    final = epoch_rows[-1]
    summary = {
        "algorithm": algorithm,
        "optimizer": optimizer_name,
        "base_lr": base_lr,
        "min_lr_ratio": min_lr_ratio,
        "epochs": epochs,
        "schedule_epochs": schedule_epochs,
        "updates": global_step,
        "full_updates": full_updates,
        "partial_updates": partial_updates,
        "partial_update_fraction": partial_updates / max(global_step, 1),
        "final_train_loss": train_loss,
        "final_train_accuracy": train_accuracy,
        "final_eval_loss": final["eval_loss"],
        "final_eval_accuracy": final["eval_accuracy"],
        "augmentation_seconds": augmentation_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "training_wall_seconds": cumulative_training_seconds,
        "evaluation_seconds": sum(float(row["eval_seconds"]) for row in epoch_rows),
        "warmup_steps": warmup_steps,
        "warmup_seconds_excluded": warmup_seconds,
        "ema_decay": ema_decay,
    }
    for target in targets:
        pct = int(round(100 * float(target)))
        summary[f"time_to_{pct}pct_seconds"] = first_target_seconds[float(target)]
        summary[f"epoch_to_{pct}pct"] = first_target_epoch[float(target)]
    return epoch_rows, summary


def parse_algorithms(value: str) -> List[str]:
    allowed = {"adam", "adan", "loss_gated_moment_continuation_adam"}
    result = [part.strip() for part in value.split(",") if part.strip()]
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"unsupported algorithms: {sorted(unknown)}")
    if not result:
        raise ValueError("at least one algorithm is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-subset", type=int, default=50000)
    parser.add_argument("--eval-subset", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--schedule-epochs", type=int, default=0,
        help="cosine horizon; zero uses --epochs",
    )
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adam-lr", type=float, default=0.0003)
    parser.add_argument("--adan-lr", type=float, default=0.0003)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=196)
    parser.add_argument(
        "--algorithms",
        default="adam,adan,loss_gated_moment_continuation_adam",
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--out", default="stage6a1_cifar10_augmented_seed40")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    schedule_epochs = args.schedule_epochs or args.epochs
    if schedule_epochs < args.epochs:
        raise ValueError("schedule epochs must be at least the number of run epochs")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema decay must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("minimum LR ratio must be in [0, 1]")
    algorithms = parse_algorithms(args.algorithms)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    )
    v081.set_seed(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "cifar10":
        x_train_raw, y_train, x_eval, y_eval = load_cifar10_raw(
            Path(args.data_dir), args.train_subset, args.eval_subset, args.seed, device
        )
    else:
        x_train_raw, y_train, x_eval, y_eval = load_fake_raw(
            args.train_subset, args.eval_subset, args.seed, device
        )
    loss_fn = nn.CrossEntropyLoss()
    model = s5a1.make_cifar_resnet18_gn().to(device)
    init_state = v4g.clone_state_dict(model)
    parameters = v012.parameter_count(model)

    warm_x = normalize_cifar(x_train_raw[: min(args.batch_size, len(x_train_raw))])
    warm_y = y_train[: len(warm_x)]
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        loss_fn(model(warm_x), warm_y).backward()
    v081.sync(device)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(init_state)

    summaries: List[dict] = []
    common = {
        "seed": args.seed,
        "dataset": args.dataset,
        "model": "CIFARResNet18GN",
        "model_parameters": parameters,
        "train_subset": args.train_subset,
        "eval_subset": args.eval_subset,
        "batch_size": args.batch_size,
        "augmentation": "random_crop_padding4_zero_then_horizontal_flip_p0.5",
        "normalization": "cifar10_mean_std",
        "schedule": "per_update_cosine",
        "candidate_mechanism": "loss_gated_moment_continuation_adam",
        "controller_frozen_from": "stage5f",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    for algorithm in algorithms:
        base_lr = args.adan_lr if algorithm == "adan" else args.adam_lr
        rows, summary = run_one(
            algorithm, base_lr, model, init_state,
            x_train_raw, y_train, x_eval, y_eval, loss_fn, device,
            args.epochs, schedule_epochs, args.batch_size, args.seed, args.eval_batch_size,
            args.min_lr_ratio, args.ema_decay, TARGET_ACCURACIES,
            args.warmup_steps,
        )
        for row in rows:
            row.update(common)
            row["base_lr"] = base_lr
        summary.update(common)
        v081.write_csv(output / f"{algorithm}_epoch_trace.csv", rows)
        summaries.append(summary)
    v081.write_csv(output / "trajectory_summary.csv", summaries)

    baselines = [row for row in summaries if row["algorithm"] in ("adam", "adan")]
    candidate = next(
        (row for row in summaries if row["algorithm"] == "loss_gated_moment_continuation_adam"),
        None,
    )
    comparisons: List[dict] = []
    if candidate is not None:
        for baseline in baselines:
            comparisons.append({
                "seed": args.seed,
                "baseline": baseline["algorithm"],
                "candidate": candidate["algorithm"],
                "baseline_eval_accuracy": baseline["final_eval_accuracy"],
                "candidate_eval_accuracy": candidate["final_eval_accuracy"],
                "accuracy_delta_pp": 100.0 * (
                    candidate["final_eval_accuracy"] - baseline["final_eval_accuracy"]
                ),
                "baseline_training_seconds": baseline["training_wall_seconds"],
                "candidate_training_seconds": candidate["training_wall_seconds"],
                "training_time_reduction": 1.0 - (
                    candidate["training_wall_seconds"] / baseline["training_wall_seconds"]
                ),
                "candidate_partial_fraction": candidate["partial_update_fraction"],
            })
    v081.write_csv(output / "paired_comparison.csv", comparisons)
    lines = [
        "Stage 6A augmented CIFAR-10 development run",
        f"Controller frozen from Stage 5F; seed={args.seed}.",
    ]
    for row in summaries:
        lines.append(
            f"{row['algorithm']}: eval={100 * row['final_eval_accuracy']:.2f}%; "
            f"train_s={row['training_wall_seconds']:.2f}; "
            f"partial={100 * row['partial_update_fraction']:.2f}%"
        )
    report = "\n".join(lines) + "\n"
    (output / "REPORT.txt").write_text(report, encoding="utf-8")
    (output / "RUN_CONFIG.txt").write_text(
        " ".join([str(value) for value in vars(args).items()]) + "\n",
        encoding="utf-8",
    )
    print("\n" + report)


if __name__ == "__main__":
    main()
