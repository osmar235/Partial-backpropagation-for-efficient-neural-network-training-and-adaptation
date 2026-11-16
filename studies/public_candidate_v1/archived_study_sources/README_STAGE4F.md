# AGRLS Stage 4F — mechanism ablation before scaling

## Why this experiment comes before CIFAR-10
The frozen AGRLS-v0.14 stochastic controller has already shown large backward/compute savings on Fashion-MNIST. Before scaling to an 11M-parameter residual network, Stage 4F tests the most obvious alternative explanation: **is the benefit mostly just reusing one gradient for two updates?**

This is a **development-only mechanism ablation** using already-seen seeds 64–72. It creates no new held-out claim and does not tune AGRLS-v0.14.

## Compared methods
1. **Adam** — one exact backward per update.
2. **Naive K=2 reuse** — one exact gradient updates Adam state once; the resulting Adam displacement is applied twice, unconditionally, with no extra forward between steps.
3. **Monotonic-gated K=2 reuse** — exact gradient -> Adam anchor -> same-batch forward; if the anchor lowers the locked-batch loss, the exact same Adam displacement is repeated once. No curvature fit.
4. **AGRLS-v0.14** — exact gradient -> Adam anchor -> same-batch forward -> 1-D information-matched curvature fit -> optional surrogate-chosen second move.

Each non-Adam mechanism is tested in two modes:
- **fixed 400 accepted updates**;
- **equal calibrated NN compute** relative to Adam.

## Interpretation
The key comparison is not merely whether all reuse methods save backwards. We want to know whether AGRLS gives a better **accuracy / backward / compute frontier** than:
- unconditional reuse, and
- a very simple monotonic-loss gate.

If the simple baselines match AGRLS, the paper should be reframed toward adaptive gradient reuse rather than claiming that the local surrogate is essential. If AGRLS is materially better, this ablation directly supports the surrogate mechanism before the CIFAR-10 scaling phase.

## Run
From Windows CMD inside the project folder:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4f_mechanism_ablation_seen_seeds_64_to_72.cmd
```

Direct command:

```cmd
python agrls_stage4f_mechanism_ablation_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --adam-updates 400 --max-equal-updates 850 --seeds 64 65 66 67 68 69 70 71 72 --device cuda --adam-lr 0.001 --out stage4f_mechanism_ablation_seen_seeds_64_to_72
```

## Outputs
- `raw_summary.csv`
- `paired_vs_adam.csv`
- `aggregate_paired_summary.csv`
- `agrls_vs_reuse_head_to_head.csv`
- fixed/equal-compute diagnostic figures
- per-seed traces and summaries
- `REPORT.txt`
