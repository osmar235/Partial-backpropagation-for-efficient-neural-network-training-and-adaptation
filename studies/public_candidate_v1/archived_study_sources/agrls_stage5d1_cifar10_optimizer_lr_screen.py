from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from third_party.Adan.adan import Adan

import agrls_stage1_mnist_fullbatch_v081 as v081
import agrls_stage4a_minibatch_locked_v012 as v012
import agrls_stage4g_forward_gated_refresh as v4g
import agrls_stage5a1_cifar10_fggr_microprobe as s5a1


def make_optimizer(name: str, model: nn.Module, lr: float):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adan":
        return Adan(
            model.parameters(), lr=lr, weight_decay=0.0, max_grad_norm=0.0,
            foreach=True, fused=False,
        )
    raise ValueError(name)


def run_one(
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
):
    model.load_state_dict(init_state)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = make_optimizer(optimizer_name, model, lr)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)
    forward_seconds = backward_seconds = optimizer_seconds = 0.0
    v081.sync(device)
    wall_start = time.perf_counter()
    for update in range(1, updates + 1):
        idx = next(stream)
        xb, yb = v012.batch_from_indices(xtr, ytr, idx, device)
        optimizer.zero_grad(set_to_none=True)
        v081.sync(device); start = time.perf_counter()
        loss = loss_fn(model(xb), yb)
        v081.sync(device); forward_seconds += time.perf_counter() - start
        v081.sync(device); start = time.perf_counter()
        loss.backward()
        v081.sync(device); backward_seconds += time.perf_counter() - start
        v081.sync(device); start = time.perf_counter()
        optimizer.step()
        v081.sync(device); optimizer_seconds += time.perf_counter() - start
        if update % 100 == 0:
            print(
                f"{optimizer_name} lr={lr:g}: completed {update}/{updates}",
                flush=True,
            )
    v081.sync(device)
    wall_seconds = time.perf_counter() - wall_start
    train_loss, train_accuracy = s5a1.evaluate_batched(
        model, xtr, ytr, loss_fn, eval_batch_size
    )
    eval_loss, eval_accuracy = s5a1.evaluate_batched(
        model, xev, yev, loss_fn, eval_batch_size
    )
    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "updates": updates,
        "final_train_loss": train_loss,
        "final_train_accuracy": train_accuracy,
        "final_eval_loss": eval_loss,
        "final_eval_accuracy": eval_accuracy,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "measured_training_seconds": forward_seconds + backward_seconds + optimizer_seconds,
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-subset", type=int, default=50000)
    parser.add_argument("--eval-subset", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adam-lrs", type=float, nargs="+", default=[0.0003, 0.001, 0.003])
    parser.add_argument("--adan-lrs", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.01])
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--out", default="stage5d1_cifar10_optimizer_lr_screen_seed10")
    args = parser.parse_args()
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

    # Neutral warm-up before the timed screen.
    warm_x, warm_y = xtr[: args.batch_size], ytr[: args.batch_size]
    model.zero_grad(set_to_none=True); loss_fn(model(warm_x), warm_y).backward()
    v081.sync(device); model.zero_grad(set_to_none=True); model.load_state_dict(init_state)

    rows: List[dict] = []
    for optimizer_name, lrs in (("adam", args.adam_lrs), ("adan", args.adan_lrs)):
        for lr in lrs:
            row = run_one(
                optimizer_name, lr, model, init_state, xtr, ytr, xev, yev,
                loss_fn, device, args.updates, args.batch_size, args.seed,
                args.eval_batch_size,
            )
            row.update({
                "seed": args.seed,
                "dataset": args.dataset,
                "model": "CIFARResNet18GN",
                "model_parameters": parameters,
                "batch_size": args.batch_size,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
            })
            rows.append(row)
    best = {}
    for optimizer_name in ("adam", "adan"):
        candidates = [row for row in rows if row["optimizer"] == optimizer_name]
        best[optimizer_name] = max(
            candidates,
            key=lambda row: (float(row["final_eval_accuracy"]), -float(row["final_eval_loss"])),
        )
    v081.write_csv(output / "optimizer_lr_screen.csv", rows)
    v081.write_csv(output / "selected_baselines.csv", list(best.values()))
    report = [
        "AGRLS Stage 5D1 - CIFAR-10 optimizer learning-rate screen\n",
        f"seed={args.seed}; gpu={rows[0]['gpu_name']}; updates={args.updates}\n\n",
        "optimizer lr eval_accuracy eval_loss wall_seconds\n",
    ]
    for row in rows:
        report.append(
            f"{row['optimizer']:>9s} {float(row['lr']):.6g} "
            f"{100 * float(row['final_eval_accuracy']):.2f}% "
            f"{float(row['final_eval_loss']):.5f} {float(row['wall_seconds']):.3f}\n"
        )
    report.append(
        f"\nselected_adam_lr={best['adam']['lr']}; "
        f"selected_adan_lr={best['adan']['lr']}\n"
    )
    (output / "REPORT.txt").write_text("".join(report), encoding="utf-8")
    print("\n" + "".join(report))


if __name__ == "__main__":
    main()
