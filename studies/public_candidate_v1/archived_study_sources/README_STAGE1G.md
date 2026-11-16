# AGRLS Stage 1G — v0.9 Deferred-Backward / Cached-Forward MVR

## Why this experiment exists
The held-out v0.8.1 batch (seeds 10–18) confirmed that the method can reduce exact gradients while preserving accuracy, but a timing audit found that the old `oracle_seconds` metric was not apples-to-apples: Adam's baseline timing included `optimizer.step()`, whereas AGRLS's exact-gradient timing excluded its Adam-state/controller work. A common calibrated cost model also showed that v0.8.1 still used more forward-equivalent neural-network compute because every accepted AGRLS update paid a standalone forward, and most later gradient refreshes recomputed the forward at the same parameter point.

v0.9 changes the *execution mechanism*, not the 1-D surrogate idea:

1. Every candidate point is evaluated with an autograd graph retained.
2. If the controller decides to continue without backpropagation, the graph is discarded.
3. If the controller decides to refresh the exact gradient at that same accepted point, the already-paid forward is **upgraded to backward**. The forward is not recomputed.
4. Adam is instrumented with separate forward, backward, and optimizer-step timings so neural-network compute can be compared fairly.

This makes the research question literal: after seeing the forward loss, decide whether the backward pass is worth paying for.

## Two controller variants
Both use the same 1-D information-matched surrogate and Adam anchoring.

- `agrls_db_legacy`: retains the old MVR threshold form, `1/(r+1)`, under the new fair calibration.
- `agrls_db_incremental`: uses the incremental deferred-backward cost. Once the current forward is already paid, refresh cost is `backward + next forward`, which equals one full-gradient/forward ratio `r`; therefore the threshold is `1/r`.

This is a development comparison on seed 0 only. Do not treat it as held-out validation.

## Run
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage1g_v09_deferred_backward_seed0.cmd
```

Equivalent direct command:
```cmd
python agrls_stage1_mnist_fullbatch_v09.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --algorithms adam agrls_db_legacy agrls_db_incremental --out stage1g_mnist_deferred_backward_seed0
```

## Send back
Zip and upload the entire folder:
`stage1g_mnist_deferred_backward_seed0`

## Primary readouts
- final evaluation accuracy
- forward passes
- backward passes
- forward-only passes
- measured forward seconds
- measured backward seconds
- `nn_compute_seconds = forward_seconds + backward_seconds`
- total wall time
- backward upgrades vs fresh full-gradient recomputations

## Smoke test status
A small fake-data CPU smoke test was run only to verify code paths. It is not scientific data.
