# Stage 8 v2: current run guide

Updated 2026-10-19. The v1 baseline screen completed and correctly stopped:
LR0.001 averaged **83.19%** final validation accuracy (82.92%,83.46%), below
the declared mean85%/minimum83% readiness thresholds. LR0.0003 averaged81.59%.
The four runs passed integrity/checkpoint review and remain unchanged in
`results/stage8A_vit_development/`. Review: `results/stage8A_review_v1/REPORT.md`.

One prospective amendment uses the same compact ViT, development split and
seeds, fixed LR0.001, **300 epochs and MixUp alpha0.2**. It retains the same
readiness thresholds. This is an adaptively revised development recipe, not
independent validation or a promise of better performance. The failed screen
and amendment must be disclosed in the eventual supplement.

Protocol: `protocol/stage8/STAGE8_PROTOCOL_v2.md`; machine settings:
`protocol/stage8/study_spec_v2.json`. No evaluation seeds have been run.

## Run in Windows CMD

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage8A_vit_development_v2.cmd
```

Stop other substantial GPU jobs first; the launcher checks startup utilization.
It runs **two full-AdamW baselines**, seeds1001/1002, 300epochs each, on the
unchanged training-only45k/validation5k split. Test images are not loaded.
If mean final accuracy>=85% AND each seed>=83%, it then runs **two LGPB
calibrations**. Otherwise it stops with a clear readiness message and retains
the screen. No automatic third recipe or evaluation follows failure.

Send **`results\stage8A_vit_development_v2`** when complete or stopped. Keep
the per-run checkpoints, configuration, epoch/step traces, summaries and GPU
telemetry, plus the baseline selection/screen/report files. Matching completed
runs are skipped on rerun; unfinished runs restart from initialization.
Changes to scientific code/settings/environment fail explicitly.

## After development review

The prepared launcher is:

```cmd
call scripts\run_stage8B_freeze_and_evaluate_v2.cmd
```

It freezes development/source/settings/profile hashes in
`protocol/stage8/FROZEN_EVALUATION_v2.json`, then runs all24 trajectories on
seeds1101--1106: full AdamW, LGPB, periodic matched to development and prefix
displacement replay. All use the same300-epoch/MixUp recipe and balanced order.
Outputs: `results/stage8B_vit_evaluation_v2/`. Report all arms, seeds, timing,
uncertainty and70/80/85% targets including failures. Criteria remain separate
from original Stage6 confirmation. Six seeds do not guarantee sufficient precision.

The repeated validation split and prior project exposure to CIFAR10 test data
must be disclosed. No Stage8 scientific result is added to the manuscript yet.
Publication packaging/public identifiers remain unfinished.

## Artificial-data diagnostics

```cmd
call scripts\run_stage8_00_sanity_v2.cmd
```

This runs CPU tests and a complete tiny fake-data development/freeze/24-arm
rehearsal under `results/stage8_sanity_recipe_v2/`. It never provides scientific
accuracy or speed evidence. See `protocol/stage8/VALIDATION_v2.md` for validation.

Read-only recomputation after real evaluation:

```cmd
"%PYTHON%" -B scripts\stage8_workflow_v2.py summarize
"%PYTHON%" -B scripts\stage8_audit_v2.py --root results\stage8B_vit_evaluation_v2 --freeze-file protocol\stage8\FROZEN_EVALUATION_v2.json
```

Executed v1 sources/protocol are retained for their saved hashes. The older
`README_STAGE8.md` commands describe that historical recipe; use the v2
launchers above for the current amendment. No old results should be deleted.
