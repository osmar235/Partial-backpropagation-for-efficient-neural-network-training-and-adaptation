# AGRLS Stage 2B — v0.10 Adam-consistent anchor development

Purpose: diagnose the Fashion-MNIST transfer failure of frozen v0.9 without changing more than one structural rule.

## What changes in v0.10
When AGRLS pays for an exact gradient, the resulting standard Adam anchor step is accepted exactly as the Adam baseline would take it. It is no longer backtracked merely because the immediate loss is non-monotone.

If that anchor is non-monotone (or its Adam direction is not locally descending under the current exact gradient), the method disables surrogate continuation for that cycle and immediately upgrades the already-computed forward at the new point to backward. This preserves the Adam trajectory without paying a duplicate forward.

Surrogate-only continuation is attempted only after a locally productive anchor.

## What is held fixed
- Fashion-MNIST
- MLP 784 -> 128 -> 64 -> 10
- 20,000 train examples; 5,000 eval examples
- 60 accepted updates
- Adam LR 0.01
- 1-D surrogate
- deferred-backward execution
- incremental MVR threshold
- no dedicated map probes
- same timing/calibration instrumentation

## Development comparison
Seed 28 only (already seen in Stage 2A):
- Adam
- v0.9 incremental (backtracking anchor)
- v0.10 Adam-consistent anchor

This is a development diagnostic, not held-out evidence.

## Run
From CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage2b_v10_fashion_seed28.cmd
```

Or directly:

```cmd
python agrls_stage2_fashion_fullbatch_v10.py --dataset fashionmnist --train-subset 20000 --eval-subset 5000 --updates 60 --seed 28 --device cuda --algorithms adam agrls_db_incremental agrls_db_adam_consistent --out stage2b_fashionmnist_v10_seed28
```

Zip `stage2b_fashionmnist_v10_seed28` and upload it for analysis.
