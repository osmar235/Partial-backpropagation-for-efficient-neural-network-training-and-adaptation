from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

import agrls_stage1_mnist_fullbatch_v081 as v081
import agrls_stage4a_minibatch_locked_v012 as v012
import agrls_stage4g_forward_gated_refresh as v4g
import agrls_stage5a1_cifar10_fggr_microprobe as s5a1
import agrls_stage5d1_cifar10_optimizer_lr_screen as s5d1


SUFFIX_PREFIXES = ("layer2.", "layer3.", "layer4.", "fc.")


def run_one(
    algorithm: str,
    optimizer_name: str,
    lr: float,
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
    eval_batch_size: int,
    ema_decay: float,
):
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
    compensated = algorithm == "loss_gated_compensated_adam"
    prefix_continuation = algorithm == "loss_gated_prefix_continuation_adam"
    moment_continuation = algorithm == "loss_gated_moment_continuation_adam"
    if compensated:
        if optimizer_name != "adam":
            raise ValueError("cadence compensation currently requires Adam")
        optimizer = torch.optim.Adam(
            [
                {"params": prefix_parameters, "lr": lr},
                {"params": suffix_parameters, "lr": lr},
            ],
            lr=lr,
        )
    else:
        optimizer = s5d1.make_optimizer(optimizer_name, model, lr)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)
    rows: List[dict] = []
    loss_ema = None
    previous_partial = False
    last_prefix_displacement = None
    full_updates = partial_updates = 0
    forward_seconds = backward_seconds = optimizer_seconds = 0.0
    v081.sync(device); wall_start = time.perf_counter()
    for update in range(1, updates + 1):
        idx = next(stream)
        xb, yb = v012.batch_from_indices(xtr, ytr, idx, device)
        optimizer.zero_grad(set_to_none=True)
        v081.sync(device); start = time.perf_counter()
        loss = loss_fn(model(xb), yb)
        v081.sync(device); elapsed_forward = time.perf_counter() - start
        loss_value = float(loss.detach().item())
        ema_before = float("nan") if loss_ema is None else loss_ema

        if algorithm in ("adam", "adan"):
            use_partial = False
            decision = "full_baseline"
        elif algorithm == "fixed_block_adam":
            use_partial = update % 2 == 0
            decision = "partial_fixed" if use_partial else "full_fixed"
        elif algorithm in (
            "loss_gated_block_adam",
            "loss_gated_compensated_adam",
            "loss_gated_prefix_continuation_adam",
            "loss_gated_moment_continuation_adam",
        ):
            eligible = loss_ema is not None and not previous_partial
            use_partial = bool(eligible and loss_value <= loss_ema)
            if use_partial:
                decision = "partial_loss_not_above_ema"
            elif previous_partial:
                decision = "full_forced_after_partial"
            elif loss_ema is None:
                decision = "full_initialize_ema"
            else:
                decision = "full_loss_above_ema"
        else:
            raise ValueError(algorithm)

        v081.sync(device); start = time.perf_counter()
        if use_partial:
            gradients = torch.autograd.grad(loss, suffix_parameters)
            for parameter, gradient in zip(suffix_parameters, gradients):
                parameter.grad = gradient
            if moment_continuation:
                for parameter in prefix_parameters:
                    parameter.grad = torch.zeros_like(parameter)
            partial_updates += 1
        else:
            loss.backward()
            full_updates += 1
        v081.sync(device); elapsed_backward = time.perf_counter() - start
        prefix_lr_multiplier = 1.0
        if compensated:
            prefix_lr_multiplier = 2.0 if (not use_partial and previous_partial) else 1.0
            optimizer.param_groups[0]["lr"] = lr * prefix_lr_multiplier
            optimizer.param_groups[1]["lr"] = lr
        v081.sync(device); start = time.perf_counter()
        prefix_before = None
        if prefix_continuation and not use_partial:
            prefix_before = [parameter.detach().clone() for parameter in prefix_parameters]
        optimizer.step()
        if prefix_continuation:
            if use_partial:
                if last_prefix_displacement is None:
                    raise RuntimeError("prefix continuation has no exact anchor")
                with torch.no_grad():
                    for parameter, displacement in zip(
                        prefix_parameters, last_prefix_displacement
                    ):
                        parameter.add_(displacement)
            else:
                last_prefix_displacement = [
                    parameter.detach().sub(before)
                    for parameter, before in zip(prefix_parameters, prefix_before)
                ]
        v081.sync(device); elapsed_optimizer = time.perf_counter() - start

        loss_ema = loss_value if loss_ema is None else (
            ema_decay * loss_ema + (1.0 - ema_decay) * loss_value
        )
        previous_partial = use_partial
        forward_seconds += elapsed_forward
        backward_seconds += elapsed_backward
        optimizer_seconds += elapsed_optimizer
        rows.append({
            "algorithm": algorithm,
            "update": update,
            "decision": decision,
            "batch_loss": loss_value,
            "loss_ema_before": ema_before,
            "loss_minus_ema": loss_value - ema_before if loss_ema is not None else float("nan"),
            "partial_update": use_partial,
            "full_updates": full_updates,
            "partial_updates": partial_updates,
            "prefix_lr_multiplier": prefix_lr_multiplier,
            "forward_seconds": elapsed_forward,
            "backward_seconds": elapsed_backward,
            "optimizer_seconds": elapsed_optimizer,
        })
        if update % 100 == 0:
            print(
                f"{algorithm}: completed {update}/{updates}; "
                f"partial={partial_updates}", flush=True
            )
    v081.sync(device); wall_seconds = time.perf_counter() - wall_start
    train_loss, train_accuracy = s5a1.evaluate_batched(
        model, xtr, ytr, loss_fn, eval_batch_size
    )
    eval_loss, eval_accuracy = s5a1.evaluate_batched(
        model, xev, yev, loss_fn, eval_batch_size
    )
    summary = {
        "algorithm": algorithm,
        "optimizer": optimizer_name,
        "lr": lr,
        "updates": updates,
        "full_updates": full_updates,
        "partial_updates": partial_updates,
        "partial_update_fraction": partial_updates / max(updates, 1),
        "final_train_loss": train_loss,
        "final_train_accuracy": train_accuracy,
        "final_eval_loss": eval_loss,
        "final_eval_accuracy": eval_accuracy,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "measured_training_seconds": forward_seconds + backward_seconds + optimizer_seconds,
        "wall_seconds": wall_seconds,
        "ema_decay": ema_decay,
        "compensated_prefix_updates": sum(
            float(row["prefix_lr_multiplier"]) > 1.0 for row in rows
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-subset", type=int, default=50000)
    parser.add_argument("--eval-subset", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--candidate-updates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adam-lr", type=float, default=0.0003)
    parser.add_argument("--adan-lr", type=float, default=0.0003)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--comparison-mode", choices=["fixed_updates", "equal_wall"],
        default="fixed_updates",
    )
    parser.add_argument("--include-compensated", action="store_true")
    parser.add_argument("--include-prefix-continuation", action="store_true")
    parser.add_argument("--include-moment-continuation", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--out", default="stage5d2_cifar10_loss_gated_seed10")
    args = parser.parse_args()
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema decay must be in [0, 1)")
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    )
    v081.set_seed(args.seed)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "cifar10":
        xtr, ytr, xev, yev = s5a1.load_cifar10_tensors(
            Path(args.data_dir), args.train_subset, args.eval_subset, args.seed, device
        )
    else:
        xtr, ytr, xev, yev = s5a1.load_fake_cifar(
            args.train_subset, args.eval_subset, args.seed, device
        )
    loss_fn = nn.CrossEntropyLoss()
    model = s5a1.make_cifar_resnet18_gn().to(device)
    init_state = v4g.clone_state_dict(model)
    parameters = v012.parameter_count(model)
    warm_x, warm_y = xtr[: args.batch_size], ytr[: args.batch_size]
    model.zero_grad(set_to_none=True); loss_fn(model(warm_x), warm_y).backward()
    v081.sync(device); model.zero_grad(set_to_none=True); model.load_state_dict(init_state)

    candidate_updates = args.candidate_updates or args.updates
    if args.include_moment_continuation:
        configs = [
            ("adam", "adam", args.adam_lr, args.updates),
            ("adan", "adan", args.adan_lr, args.updates),
            (
                "loss_gated_prefix_continuation_adam", "adam", args.adam_lr,
                args.updates,
            ),
            (
                "loss_gated_moment_continuation_adam", "adam", args.adam_lr,
                candidate_updates,
            ),
        ]
        candidate_name = "loss_gated_moment_continuation_adam"
    elif args.include_prefix_continuation:
        configs = [
            ("adam", "adam", args.adam_lr, args.updates),
            ("adan", "adan", args.adan_lr, args.updates),
            ("loss_gated_block_adam", "adam", args.adam_lr, args.updates),
            (
                "loss_gated_prefix_continuation_adam", "adam", args.adam_lr,
                candidate_updates,
            ),
        ]
        candidate_name = "loss_gated_prefix_continuation_adam"
    elif args.include_compensated:
        configs = [
            ("adam", "adam", args.adam_lr, args.updates),
            ("adan", "adan", args.adan_lr, args.updates),
            ("loss_gated_block_adam", "adam", args.adam_lr, args.updates),
            (
                "loss_gated_compensated_adam", "adam", args.adam_lr,
                candidate_updates,
            ),
        ]
        candidate_name = "loss_gated_compensated_adam"
    else:
        configs = [
            ("adam", "adam", args.adam_lr, args.updates),
            ("adan", "adan", args.adan_lr, args.updates),
            ("fixed_block_adam", "adam", args.adam_lr, args.updates),
            ("loss_gated_block_adam", "adam", args.adam_lr, candidate_updates),
        ]
        candidate_name = "loss_gated_block_adam"
    summaries = []
    for algorithm, optimizer_name, lr, run_updates in configs:
        rows, summary = run_one(
            algorithm, optimizer_name, lr, model, init_state,
            xtr, ytr, xev, yev, loss_fn, device, run_updates,
            args.batch_size, args.seed, args.eval_batch_size, args.ema_decay,
        )
        common = {
            "seed": args.seed,
            "dataset": args.dataset,
            "model": "CIFARResNet18GN",
            "model_parameters": parameters,
            "batch_size": args.batch_size,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        summary.update(common)
        for row in rows: row.update(common)
        v081.write_csv(output / f"{algorithm}_trace.csv", rows)
        summaries.append(summary)
    v081.write_csv(output / "trajectory_summary.csv", summaries)
    strongest_baseline = max(
        (row for row in summaries if row["algorithm"] in ("adam", "adan")),
        key=lambda row: float(row["final_eval_accuracy"]),
    )
    candidate = next(row for row in summaries if row["algorithm"] == candidate_name)
    accuracy_delta_pp = 100 * (
        candidate["final_eval_accuracy"] - strongest_baseline["final_eval_accuracy"]
    )
    wall_reduction = 1.0 - candidate["wall_seconds"] / strongest_baseline["wall_seconds"]
    if args.comparison_mode == "fixed_updates":
        passes = accuracy_delta_pp >= -1.0 and wall_reduction >= 0.05
    else:
        passes = accuracy_delta_pp >= -1.0 and wall_reduction >= 0.0
    comparison = {
        "strongest_baseline": strongest_baseline["algorithm"],
        "accuracy_delta_percentage_points": accuracy_delta_pp,
        "wall_time_reduction": wall_reduction,
        "development_criteria_pass": passes,
        "seed": args.seed,
        "ema_decay": args.ema_decay,
        "comparison_mode": args.comparison_mode,
        "baseline_updates": args.updates,
        "candidate_updates": candidate_updates,
    }
    v081.write_csv(output / "paired_comparison.csv", [comparison])
    report = [
        "AGRLS Stage 5D2 - loss-gated exact blockwise Adam\n",
        f"seed={args.seed}; gpu={summaries[0]['gpu_name']}; "
        f"baseline_updates={args.updates}; candidate_updates={candidate_updates}\n\n",
        "algorithm updates eval_accuracy full partial partial_rate wall_seconds\n",
    ]
    for row in summaries:
        report.append(
            f"{row['algorithm']:>24s} {int(row['updates'])} "
            f"{100 * float(row['final_eval_accuracy']):.2f}% "
            f"{int(row['full_updates'])} {int(row['partial_updates'])} "
            f"{float(row['partial_update_fraction']):.4f} {float(row['wall_seconds']):.3f}\n"
        )
    report.extend([
        f"\nstrongest_baseline={strongest_baseline['algorithm']}\n",
        f"candidate_accuracy_delta_pp={accuracy_delta_pp:.3f}\n",
        f"candidate_wall_time_reduction={wall_reduction:.4f}\n",
        f"comparison_mode={args.comparison_mode}\n",
        f"development_criteria_pass={passes}\n",
    ])
    (output / "REPORT.txt").write_text("".join(report), encoding="utf-8")
    print("\n" + "".join(report))


if __name__ == "__main__":
    main()
