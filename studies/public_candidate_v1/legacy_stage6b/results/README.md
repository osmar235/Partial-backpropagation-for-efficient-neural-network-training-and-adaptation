# Results organization

## `stage5f3/`

Frozen short-horizon held-out results for seeds 31-39. This block failed its
pre-specified aggregate accuracy gate and is retained as negative evidence.

## `stage6_development/`

Seed-40 augmented development runs, reverse-order crossover, timing
qualification runs, Adan learning-rate screen, and the final tuned-Adan
development run. Seed 40 is not confirmatory evidence.

## `stage6b/`

The final held-out confirmatory records for seeds 41-49. Each seed directory
contains per-method epoch traces, run configuration, trajectory summaries, and
paired comparisons. `stage6b_heldout_aggregate/` contains the paired table,
machine-readable statistics, and human-readable report produced by
`scripts/analyze_stage6b.py`.

No model checkpoints or CIFAR-10 images are included. The reported statistics
are reproducible from the included trajectory summaries.

