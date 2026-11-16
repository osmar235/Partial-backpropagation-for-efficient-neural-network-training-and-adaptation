# AGRLS Stage 2A — Frozen v0.9 transfer to Fashion-MNIST

This package tests the already-frozen AGRLS-v0.9 incremental deferred-backward controller on a new dataset without changing its optimization logic.

## Scientific purpose
Stage 1H passed its preregistered MNIST held-out criteria. Stage 2A asks whether the same mechanism transfers to Fashion-MNIST before we introduce mini-batch stochasticity, which would change the semantics of the local surrogate.

## Frozen setup
- Dataset: Fashion-MNIST
- Model: MLP 784 -> 128 -> 64 -> 10
- Train subset: 20,000
- Evaluation subset: 5,000
- Accepted updates: 60
- Algorithms: Adam and frozen AGRLS-v0.9 incremental
- Seeds: 28-36
- Execution order: alternated by seed
- No change to AGRLS threshold, surrogate, retry logic, Adam anchor LR, or deferred-backward mechanism

## Primary criteria (declared before these runs)
1. median evaluation-accuracy deficit vs Adam <= 1.0 percentage point
2. median backward-pass reduction vs Adam >= 10%
3. median common calibrated pure NN-compute reduction > 0%

Direct implementation-path NN time and wall time remain secondary because the current research implementation still contains AGRLS-specific vector/plumbing overhead.

## Run
In Windows CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage2a_v09_fashion_seeds_28_to_36.cmd
```

Or directly:

```cmd
python agrls_stage2a_v09_fashion_heldout_batch.py --dataset fashionmnist --train-subset 20000 --eval-subset 5000 --updates 60 --seeds 28 29 30 31 32 33 34 35 36 --device cuda --out stage2a_fashionmnist_fullbatch_v09_seeds_28_to_36
```

When finished, ZIP the entire `stage2a_fashionmnist_fullbatch_v09_seeds_28_to_36` folder and upload it.

## Software validation
The package was syntax-checked and smoke-tested only with fake CPU data. No Fashion-MNIST scientific result was generated during package construction.
