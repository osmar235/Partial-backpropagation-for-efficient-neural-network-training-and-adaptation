from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 20)))
    parser.add_argument("--out", default="stage5d4_cifar10_heldout_summary")
    parser.add_argument("--run-prefix", default="stage5d4_cifar10_heldout_seed")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        path = Path(args.root) / f"{args.run_prefix}{seed}" / "paired_comparison.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        rows.append({
            key: int(value) if key == "seed" else (
                float(value) if key not in ("algorithm_order", "strongest_baseline") else value
            )
            for key, value in row.items()
        })
    median = lambda key: statistics.median(float(row[key]) for row in rows)
    fixed_accuracy_passes = sum(row["fixed_accuracy_delta_pp"] >= -1.0 for row in rows)
    fixed_speed_passes = sum(row["fixed_wall_reduction"] >= 0.05 for row in rows)
    equal_accuracy_passes = sum(row["equal_wall_accuracy_delta_pp"] >= -1.0 for row in rows)
    equal_speed_passes = sum(row["equal_wall_reduction"] >= 0.0 for row in rows)
    aggregate = {
        "seed_count": len(rows),
        "seeds": ";".join(str(seed) for seed in args.seeds),
        "median_fixed_accuracy_delta_pp": median("fixed_accuracy_delta_pp"),
        "median_fixed_wall_reduction": median("fixed_wall_reduction"),
        "median_equal_wall_accuracy_delta_pp": median("equal_wall_accuracy_delta_pp"),
        "median_equal_wall_reduction": median("equal_wall_reduction"),
        "fixed_accuracy_pass_seed_count": fixed_accuracy_passes,
        "fixed_speed_pass_seed_count": fixed_speed_passes,
        "equal_accuracy_pass_seed_count": equal_accuracy_passes,
        "equal_speed_pass_seed_count": equal_speed_passes,
    }
    aggregate["fixed_mode_pass"] = (
        aggregate["median_fixed_accuracy_delta_pp"] >= -1.0
        and aggregate["median_fixed_wall_reduction"] >= 0.05
        and fixed_accuracy_passes >= 6 and fixed_speed_passes >= 6
    )
    aggregate["equal_wall_mode_pass"] = (
        aggregate["median_equal_wall_accuracy_delta_pp"] >= 0.0
        and aggregate["median_equal_wall_reduction"] >= 0.0
        and equal_accuracy_passes >= 6 and equal_speed_passes >= 6
    )
    aggregate["frozen_aggregate_pass"] = (
        aggregate["fixed_mode_pass"] and aggregate["equal_wall_mode_pass"]
    )
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate)); writer.writeheader(); writer.writerow(aggregate)
    report = (
        "AGRLS held-out CIFAR-10 aggregate\n"
        f"seeds={aggregate['seeds']}\n"
        f"median_fixed_accuracy_delta_pp={aggregate['median_fixed_accuracy_delta_pp']:.3f}\n"
        f"median_fixed_wall_reduction={aggregate['median_fixed_wall_reduction']:.4f}\n"
        f"fixed_accuracy_passes={fixed_accuracy_passes}/{len(rows)}\n"
        f"fixed_speed_passes={fixed_speed_passes}/{len(rows)}\n"
        f"median_equal_accuracy_delta_pp={aggregate['median_equal_wall_accuracy_delta_pp']:.3f}\n"
        f"median_equal_wall_reduction={aggregate['median_equal_wall_reduction']:.4f}\n"
        f"equal_accuracy_passes={equal_accuracy_passes}/{len(rows)}\n"
        f"equal_speed_passes={equal_speed_passes}/{len(rows)}\n"
        f"fixed_mode_pass={aggregate['fixed_mode_pass']}\n"
        f"equal_wall_mode_pass={aggregate['equal_wall_mode_pass']}\n"
        f"frozen_aggregate_pass={aggregate['frozen_aggregate_pass']}\n"
    )
    (output / "REPORT.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
