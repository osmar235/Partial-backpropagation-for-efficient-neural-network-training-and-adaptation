from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "stage6b"
SEEDS = list(range(41, 50))
T_CRIT_95_DF8 = 2.306004135204166
CANDIDATE = "loss_gated_moment_continuation_adam"


def interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    half = T_CRIT_95_DF8 * se
    return mean, mean - half, mean + half


rows = []
paired = []
for seed in SEEDS:
    path = ROOT / f"stage6b_heldout_seed{seed}" / "trajectory_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        by_method = {row["algorithm"]: row for row in csv.DictReader(handle)}
    if set(by_method) != {"adam", "adan", CANDIDATE}:
        raise RuntimeError(f"incomplete seed {seed}: {sorted(by_method)}")
    a, d, c = by_method["adam"], by_method["adan"], by_method[CANDIDATE]
    row = {
        "seed": seed,
        "adam_accuracy": float(a["final_eval_accuracy"]),
        "candidate_accuracy": float(c["final_eval_accuracy"]),
        "adan_accuracy": float(d["final_eval_accuracy"]),
        "candidate_minus_adam_pp": 100 * (float(c["final_eval_accuracy"]) - float(a["final_eval_accuracy"])),
        "candidate_minus_adan_pp": 100 * (float(c["final_eval_accuracy"]) - float(d["final_eval_accuracy"])),
        "adam_seconds": float(a["training_wall_seconds"]),
        "candidate_seconds": float(c["training_wall_seconds"]),
        "adan_seconds": float(d["training_wall_seconds"]),
        "candidate_adam_time_ratio": float(c["training_wall_seconds"]) / float(a["training_wall_seconds"]),
        "candidate_adan_time_ratio": float(c["training_wall_seconds"]) / float(d["training_wall_seconds"]),
        "candidate_partial_fraction": float(c["partial_update_fraction"]),
        "adam_time_to_80": float(a["time_to_80pct_seconds"]),
        "candidate_time_to_80": float(c["time_to_80pct_seconds"]),
        "adan_time_to_80": float(d["time_to_80pct_seconds"]),
    }
    paired.append(row)
    rows.extend(by_method.values())

acc = [row["candidate_minus_adam_pp"] for row in paired]
acc_mean, acc_lo, acc_hi = interval(acc)
logs = [math.log(row["candidate_adam_time_ratio"]) for row in paired]
log_mean, log_lo, log_hi = interval(logs)
ratio, ratio_lo, ratio_hi = math.exp(log_mean), math.exp(log_lo), math.exp(log_hi)
adan_acc = [row["candidate_minus_adan_pp"] for row in paired]
adan_acc_mean, adan_acc_lo, adan_acc_hi = interval(adan_acc)
adan_logs = [math.log(row["candidate_adan_time_ratio"]) for row in paired]
adan_log_mean, adan_log_lo, adan_log_hi = interval(adan_logs)
adan_ratio, adan_ratio_lo, adan_ratio_hi = map(math.exp, (adan_log_mean, adan_log_lo, adan_log_hi))
tt80_logs = [math.log(row["candidate_time_to_80"] / row["adam_time_to_80"]) for row in paired]
tt80_mean, tt80_lo, tt80_hi = interval(tt80_logs)

summary = {
    "n_seeds": len(SEEDS),
    "accuracy_delta_pp_mean": acc_mean,
    "accuracy_delta_pp_ci95": [acc_lo, acc_hi],
    "accuracy_noninferiority_margin_pp": -1.0,
    "accuracy_noninferiority_pass": acc_lo > -1.0,
    "candidate_adam_time_ratio_geomean": ratio,
    "candidate_adam_time_ratio_ci95": [ratio_lo, ratio_hi],
    "speed_claim_pass": ratio_hi < 1.0,
    "candidate_adam_wall_reduction_percent": 100 * (1 - ratio),
    "candidate_minus_adan_pp_mean": adan_acc_mean,
    "candidate_minus_adan_pp_ci95": [adan_acc_lo, adan_acc_hi],
    "candidate_adan_time_ratio_geomean": adan_ratio,
    "candidate_adan_time_ratio_ci95": [adan_ratio_lo, adan_ratio_hi],
    "candidate_adan_wall_reduction_percent": 100 * (1 - adan_ratio),
    "candidate_adam_time_to_80_ratio_geomean": math.exp(tt80_mean),
    "candidate_adam_time_to_80_ratio_ci95": [math.exp(tt80_lo), math.exp(tt80_hi)],
    "candidate_partial_fraction_mean": statistics.mean(row["candidate_partial_fraction"] for row in paired),
    "seeds_within_1pp_of_adam": sum(value >= -1.0 for value in acc),
    "seeds_faster_than_adam": sum(row["candidate_adam_time_ratio"] < 1 for row in paired),
}

out = ROOT / "stage6b_heldout_aggregate"
out.mkdir(exist_ok=True)
with (out / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
    writer.writeheader(); writer.writerows(paired)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
report = f"""# Stage 6B held-out confirmatory results

Seeds 41-49, frozen controller and protocol.

- Candidate minus Adam accuracy: {acc_mean:+.3f} pp (95% CI {acc_lo:+.3f} to {acc_hi:+.3f}).
- Preregistered -1 pp non-inferiority test: **{'PASS' if summary['accuracy_noninferiority_pass'] else 'FAIL'}**.
- Candidate/Adam wall-time ratio: {ratio:.4f} (95% CI {ratio_lo:.4f} to {ratio_hi:.4f}), or {100*(1-ratio):.2f}% lower wall time.
- Preregistered speed test: **{'PASS' if summary['speed_claim_pass'] else 'FAIL'}**.
- Per-seed counts: {summary['seeds_within_1pp_of_adam']}/9 within 1 pp; {summary['seeds_faster_than_adam']}/9 faster.
- Candidate minus tuned Adan accuracy: {adan_acc_mean:+.3f} pp (95% CI {adan_acc_lo:+.3f} to {adan_acc_hi:+.3f}).
- Candidate/tuned-Adan wall-time ratio: {adan_ratio:.4f} (95% CI {adan_ratio_lo:.4f} to {adan_ratio_hi:.4f}), or {100*(1-adan_ratio):.2f}% lower wall time.
- Candidate/Adam time-to-80% ratio: {math.exp(tt80_mean):.4f} (95% CI {math.exp(tt80_lo):.4f} to {math.exp(tt80_hi):.4f}).
- Mean partial-update fraction: {100*summary['candidate_partial_fraction_mean']:.2f}%.
"""
(out / "REPORT.md").write_text(report, encoding="utf-8")
print(report)
