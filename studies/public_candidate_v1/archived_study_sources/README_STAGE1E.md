# AGRLS Stage 1E — held-out full-batch MNIST validation

This package freezes AGRLS-v0.8 exactly as used in Stage 1D and evaluates seeds 1–9 without additional tuning.

## Scientific question
Can the information-matched 1-D Adam-anchored probe-free MVR controller retain near-Adam accuracy while reducing exact-gradient queries on held-out neural-network trajectories?

## Primary comparison
- Adam
- frozen AGRLS-v0.8 (`agrls_adam_1d_pf_mvr`)

Momentum is retained only as a contextual baseline.

Algorithm execution order is counterbalanced across seeds to reduce systematic GPU timing-order bias. Optimizer logic is unchanged.

## Preregistered supportive criterion
At the aggregate level:
1. median v0.8 evaluation accuracy no more than 1.0 percentage point below median Adam;
2. median v0.8 exact-gradient calls at least 10% lower than Adam;
3. median v0.8 measured oracle time lower than Adam.

Wall-clock time is reported but is secondary because Stage 1D showed non-oracle Python/vector-management overhead on this small MLP.

## Run on Osmar's Windows machine
Extract all files into:

`C:\Users\osmar\projects\Projeto_AI_Fronteira`

Then double-click / run:

`run_stage1e_v08_heldout_seeds_1_to_9.cmd`

Or from an activated CMD:

```cmd
python agrls_stage1e_heldout_batch.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seeds 1 2 3 4 5 6 7 8 9 --device cuda --out stage1e_mnist_fullbatch_v08_heldout_seeds_1_to_9
```

When complete, ZIP the entire folder:

`stage1e_mnist_fullbatch_v08_heldout_seeds_1_to_9`

and upload it to ChatGPT.

## Main aggregate outputs
- `heldout_raw_summary.csv`
- `heldout_aggregate_summary.csv`
- `heldout_paired_vs_adam.csv`
- `heldout_preregistered_decision.csv`
- `heldout_accuracy_vs_gradient_reduction.png`

Do not change v0.8 settings after inspecting individual seed results. Aggregate first.
