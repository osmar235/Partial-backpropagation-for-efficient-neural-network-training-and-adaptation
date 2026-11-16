# Matched BF16 quality and complete adaptation cost

The next study compares the qualified reference attention implementation under
FP32/math, BF16/math and BF16/automatic execution. It tests practical validation
quality and total adaptation time. Earlier numerical gates remain unchanged.

Run in Windows CMD with the existing project environment:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_bf16_continuation.cmd
```

Keep other GPU jobs stopped and the machine awake. Allow tens of minutes and
several GB of free disk space. The fixed design contains 36 trajectories:
two trained checkpoints, three precision policies, six counterbalanced sessions,
five epochs per trajectory. This gives 31680 measured training updates plus
108 discarded warmup updates. Timing repeats reuse the same two model seeds.

All arms use fresh AdamW state, all parameters trainable, the existing
augmentation/MixUp/LR recipe, and explicit FP32 cross-entropy outside autocast.
This is adaptation from archived weights; the original optimizer history is
reset equally. Common FP32 validation isolates training effects; a final
own-policy validation also checks deployment precision. No test set is used.

Quality must remain within 0.30 percentage points of the qualified paired FP32
reference for both evaluation modes, every session and both checkpoints. The
reference must also meet the frozen readiness checks against its source and
the previous full adaptation. Timing requires positive complete-time savings
in all six sessions per source, together with the predeclared adjusted direction
test. There is no newly chosen 5% or 10% minimum for this study. Report the
observed effect size; historical failures stay failed. A screen pass does not
establish population noninferiority, novelty or generality.

The clock includes data/model setup, warmup/restoration, training/augmentation,
epoch validations, final deployment validation and checkpoint/logit/trace writing.
Imports, provenance checks, idle preflight and post-run audits are excluded.
Profiler dispatch is recorded during warmup; automatic does not imply flash.

The launcher audits all completed checkpoints before printing COMPLETE. It
reconstructs streams and rescoring results without training. Reply “done” when
finished, or send `results/bf16_continuation_v1/` for review. On interruption,
rerun the same command: verified completed trajectories are skipped, interrupted
attempts remain visible, and fragmented timing sessions cannot qualify.

Optional audit of completed results (inference only, no training):

```cmd
"%PYTHON%" -B scripts\review_bf16_continuation.py results\bf16_continuation_v1
```

Protocol/spec/freeze: `protocol/bf16_continuation/`.
Preparation: `results/bf16_continuation_preparation_v1/REPORT.md`.
Organized source: `paper_support/bf16_continuation_addendum_v1/`.
No full adaptation study is run during preparation. Source-checkpoint inference
and tiny artificial CPU/CUDA tests are recorded separately. No installation is
required. The source capsule retains local absolute provenance and is not yet a
portable public release.
