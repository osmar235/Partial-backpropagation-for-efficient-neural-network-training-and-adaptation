# Stage11A: explore small-to-large model growth

The author approved idea1 first on 2026-10-22. This first step checks whether
the narrow full-resolution ViT can expand to the native full width without
changing its predictions, remain trainable, and offer meaningful operation-cost
headroom. It uses copies of the completed Stage10A checkpoints.

Run in Windows CMD after stopping other GPU work:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage11A_growth_feasibility.cmd
```

Send `results\stage11A_growth_feasibility_v1` when complete, or say "done".
This is a short diagnostic with validation checks and disposable update probes,
not another100-epoch sweep. Expect a few minutes, depending on startup and device
state. Reserve about0.2GB for audit checkpoints and retain incomplete attempts.

The launcher checks:

1. Width96/MLP384 -> width192/MLP768 expansion, with three attention heads and
   full32x32 input throughout. CPU/CUDA mechanics tests run first.
2. Prediction/logit preservation on the original5,000-example validation set for
   both already-seen seeds1401/1402. No test dataset is constructed.
3. Eight disposable full-gradient updates per model/seed, checking finite Adam
   states, all parameter counters and divergence of duplicated units.
4. Actual full-update operation costs for small, expanded and ordinary large
   models, with matching batches and balanced measurement order. Growth and cold
   optimizer-allocation costs are also recorded.

All diagnostic paths use fresh AdamW. Prediction preservation does not imply
identical dropout masks or optimizer trajectories. A later training pilot must
control resets and measure whether accuracy recovery consumes the early saving.

Fixed feasibility requirements in both seeds: preserved logits and predictions,
successful trainability checks, at least15% shorter small-model mean operation
time than ordinary large, and expanded/ordinary-large mean time ratio <=1.10.
A pass permits discussion of a pilot; it does not demonstrate training efficiency.
No pilot is launched automatically. Current controls use strict deterministic
FP32/math attention; deployment claims later require optimized baseline checks.

Outputs live in a numbered attempt within the result directory, including
REPORT.md, audit.json, summary.csv, per-example preservation checks, operation
timings and disposable checkpoints. COMPLETE.json identifies the sealed attempt.
Rerun the same command after an interruption: completed matching outputs verify
and skip; incomplete diagnostics restart into a new attempt without deletion.

Protocol: `protocol/stage11/GROWTH_FEASIBILITY_v1.md`.
Freeze: `protocol/stage11/FROZEN_GROWTH_FEASIBILITY_v1.json`.
Validation: `protocol/stage11/VALIDATION_GROWTH_FEASIBILITY_v1.md`.
Current master: `adaptive_gradient_refresh_project_master_v7.1.md`.

All previous results and negative outcomes remain preserved. The other proposed
directions remain queued for discussion. Publication remains on hold.
