"""Re-analyze ARCHIVED Stage 6B traces only; no GPU or training required."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/stage6b")
    p.add_argument("--out", default="results/stage7A_original_audit")
    args = p.parse_args()
    root, out = Path(args.results_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(root.glob("stage6b_heldout_seed*"))
    if not seed_dirs: raise FileNotFoundError(f"No Stage 6B seed directories in {root}")
    rows, components = [], []
    for folder in seed_dirs:
        seed = int(folder.name.rsplit("seed", 1)[1])
        if not 41 <= seed <= 49:
            raise ValueError(f"unexpected seed {seed}; this audit must analyze only Stage6B 41-49")
        for alg in ("adam", "loss_gated_moment_continuation_adam", "adan"):
            tracepath = folder / f"{alg}_epoch_trace.csv"
            if not tracepath.exists(): raise FileNotFoundError(tracepath)
            trace = records(tracepath)
            if len(trace) != 30: raise ValueError(f"{tracepath}: need 30 epochs, found {len(trace)}")
            for r in trace:
                full = int(r["epoch_full_updates"]); partial = int(r["epoch_partial_updates"])
                rows.append({"seed": seed, "algorithm": alg,
                             "epoch": int(r["epoch"]), "full_updates": full,
                             "partial_updates": partial,
                             "epoch_q": partial / (full + partial),
                             "train_loss": float(r["train_augmented_loss"]),
                             "eval_accuracy": float(r["eval_accuracy"]),
                             "epoch_wall_seconds": float(r["epoch_training_seconds"])})
        sums = records(folder / "trajectory_summary.csv")
        for r in sums:
            components.append({"seed": seed, "algorithm": r["algorithm"],
                               "total_wall_s": float(r["training_wall_seconds"]),
                               "augmentation_s": float(r["augmentation_seconds"]),
                               "forward_s": float(r["forward_seconds"]),
                               "backward_s": float(r["backward_seconds"]),
                               "optimizer_s": float(r["optimizer_seconds"]),
                               "unattributed_sync_loop_s": float(r["training_wall_seconds"]) -
                                   sum(float(r[c]) for c in ("augmentation_seconds", "forward_seconds", "backward_seconds", "optimizer_seconds")),
                               "partial_fraction": float(r["partial_update_fraction"]),
                               "accuracy": float(r["final_eval_accuracy"])})
    if len(seed_dirs) != 9: raise ValueError(f"Need all nine seeds; saw {len(seed_dirs)}")
    write(out / "archived_epoch_partial_q_by_seed.csv", rows)
    write(out / "archived_component_timing_by_seed.csv", components)
    trend = []
    cand = [x for x in rows if x["algorithm"] == "loss_gated_moment_continuation_adam"]
    for epoch in range(1, 31):
        sub = [x for x in cand if x["epoch"] == epoch]
        trend.append({"epoch": epoch, "mean_q": statistics.mean(x["epoch_q"] for x in sub),
                      "median_q": statistics.median(x["epoch_q"] for x in sub),
                      "min_q": min(x["epoch_q"] for x in sub),
                      "max_q": max(x["epoch_q"] for x in sub),
                      "mean_loss": statistics.mean(x["train_loss"] for x in sub)})
    write(out / "archived_epoch_q_summary.csv", trend)
    summary = {"seeds": sorted(int(f.name.rsplit("seed",1)[1]) for f in seed_dirs),
               "candidate_mean_q": statistics.mean(float(r["partial_fraction"]) for r in components if r["algorithm"] == "loss_gated_moment_continuation_adam"),
               "note": "No new experiments. Total component times do NOT isolate full vs partial backward; Stage7B instrumentation does."}
    (out / "audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4.5))
        for r in range(41,50):
            tr = sorted((x for x in cand if x['seed']==r),key=lambda x:x['epoch'])
            plt.plot([x['epoch'] for x in tr],[x['epoch_q'] for x in tr], alpha=.25, linewidth=.8)
        plt.plot([x['epoch'] for x in trend],[x['mean_q'] for x in trend], linewidth=2.6, label="mean over nine seeds")
        plt.xlabel("Epoch"); plt.ylabel("Partial-update fraction")
        plt.title("Frozen Stage 6B: loss-gated partial-update frequency")
        plt.ylim(0,.60); plt.legend(loc="best"); plt.tight_layout()
        plt.savefig(out / "archived_epoch_partial_fraction.png", dpi=220); plt.close()
    except ImportError:
        print('Matplotlib unavailable: CSV outputs remain complete')
    print(json.dumps(summary,indent=2)); print('Output:',out)

if __name__ == '__main__': main()
