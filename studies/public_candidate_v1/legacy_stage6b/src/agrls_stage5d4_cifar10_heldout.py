from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch
from torch import nn

import agrls_stage1_mnist_fullbatch_v081 as v081
import agrls_stage4a_minibatch_locked_v012 as v012
import agrls_stage4g_forward_gated_refresh as v4g
import agrls_stage5a1_cifar10_fggr_microprobe as s5a1
import agrls_stage5d2_cifar10_loss_gated_block_adam as s5d2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-subset", type=int, default=50000)
    parser.add_argument("--eval-subset", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--baseline-updates", type=int, default=500)
    parser.add_argument("--equal-wall-updates", type=int, default=570)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adam-lr", type=float, default=0.0003)
    parser.add_argument("--adan-lr", type=float, default=0.0003)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--candidate-mechanism",
        choices=[
            "loss_gated_block_adam",
            "loss_gated_prefix_continuation_adam",
            "loss_gated_moment_continuation_adam",
        ],
        default="loss_gated_block_adam",
    )
    parser.add_argument(
        "--algorithm-order", choices=["baseline_first", "candidate_first"],
        required=True,
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    )
    v081.set_seed(args.seed)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xev, yev = s5a1.load_cifar10_tensors(
        Path(args.data_dir), args.train_subset, args.eval_subset, args.seed, device
    )
    loss_fn = nn.CrossEntropyLoss()
    model = s5a1.make_cifar_resnet18_gn().to(device)
    init_state = v4g.clone_state_dict(model)
    parameters = v012.parameter_count(model)
    warm_x, warm_y = xtr[: args.batch_size], ytr[: args.batch_size]
    model.zero_grad(set_to_none=True); loss_fn(model(warm_x), warm_y).backward()
    v081.sync(device); model.zero_grad(set_to_none=True); model.load_state_dict(init_state)

    configs = [
        ("adam", "adam", "adam", args.adam_lr, args.baseline_updates),
        ("adan", "adan", "adan", args.adan_lr, args.baseline_updates),
        (
            "loss_gated_fixed", args.candidate_mechanism, "adam",
            args.adam_lr, args.baseline_updates,
        ),
        (
            "loss_gated_equal_wall", args.candidate_mechanism, "adam",
            args.adam_lr, args.equal_wall_updates,
        ),
    ]
    if args.algorithm_order == "candidate_first":
        configs = list(reversed(configs))
    summaries = []
    for label, mechanism, optimizer_name, lr, updates in configs:
        rows, summary = s5d2.run_one(
            mechanism, optimizer_name, lr, model, init_state,
            xtr, ytr, xev, yev, loss_fn, device, updates,
            args.batch_size, args.seed, args.eval_batch_size, args.ema_decay,
        )
        summary["algorithm"] = label
        common = {
            "seed": args.seed,
            "dataset": "cifar10",
            "model": "CIFARResNet18GN",
            "model_parameters": parameters,
            "batch_size": args.batch_size,
            "algorithm_order": args.algorithm_order,
            "candidate_mechanism": args.candidate_mechanism,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        summary.update(common)
        for row in rows:
            row["algorithm"] = label
            row.update(common)
        v081.write_csv(output / f"{label}_trace.csv", rows)
        summaries.append(summary)
    v081.write_csv(output / "trajectory_summary.csv", summaries)
    adam = next(row for row in summaries if row["algorithm"] == "adam")
    adan = next(row for row in summaries if row["algorithm"] == "adan")
    strongest = max((adam, adan), key=lambda row: float(row["final_eval_accuracy"]))
    fixed = next(row for row in summaries if row["algorithm"] == "loss_gated_fixed")
    equal = next(row for row in summaries if row["algorithm"] == "loss_gated_equal_wall")
    comparison = {
        "seed": args.seed,
        "algorithm_order": args.algorithm_order,
        "strongest_baseline": strongest["algorithm"],
        "adam_eval_accuracy": adam["final_eval_accuracy"],
        "adan_eval_accuracy": adan["final_eval_accuracy"],
        "fixed_eval_accuracy": fixed["final_eval_accuracy"],
        "equal_wall_eval_accuracy": equal["final_eval_accuracy"],
        "fixed_accuracy_delta_pp": 100 * (
            fixed["final_eval_accuracy"] - strongest["final_eval_accuracy"]
        ),
        "equal_wall_accuracy_delta_pp": 100 * (
            equal["final_eval_accuracy"] - strongest["final_eval_accuracy"]
        ),
        "fixed_wall_reduction": 1.0 - fixed["wall_seconds"] / strongest["wall_seconds"],
        "equal_wall_reduction": 1.0 - equal["wall_seconds"] / strongest["wall_seconds"],
        "fixed_partial_fraction": fixed["partial_update_fraction"],
        "equal_wall_partial_fraction": equal["partial_update_fraction"],
    }
    v081.write_csv(output / "paired_comparison.csv", [comparison])
    report = (
        "AGRLS frozen held-out loss-gated optimizer comparison\n"
        f"candidate_mechanism={args.candidate_mechanism}\n"
        f"seed={args.seed}; order={args.algorithm_order}; strongest={strongest['algorithm']}\n"
        f"adam_accuracy={100 * adam['final_eval_accuracy']:.2f}%; wall={adam['wall_seconds']:.3f}\n"
        f"adan_accuracy={100 * adan['final_eval_accuracy']:.2f}%; wall={adan['wall_seconds']:.3f}\n"
        f"fixed_accuracy={100 * fixed['final_eval_accuracy']:.2f}%; "
        f"delta_pp={comparison['fixed_accuracy_delta_pp']:.3f}; "
        f"wall_reduction={comparison['fixed_wall_reduction']:.4f}\n"
        f"equal_accuracy={100 * equal['final_eval_accuracy']:.2f}%; "
        f"delta_pp={comparison['equal_wall_accuracy_delta_pp']:.3f}; "
        f"wall_reduction={comparison['equal_wall_reduction']:.4f}\n"
    )
    (output / "REPORT.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
