# Stage 8 second architecture extension

**Historical v1 guide. Current amendment (2026-10-19): use `README_STAGE8_V2.md`
and `scripts/run_stage8A_vit_development_v2.cmd`.** V1 completed four baseline
runs but failed readiness (selected mean83.19%, required85%); no calibration
or evaluation followed. All v1 sources/protocol/raw results remain preserved.
The instructions below document the original recipe and are superseded for
new training by the bounded prospective v2 amendment.

Prepared 2026-10-18 after the author's decision to strengthen the submission.
Stage 6/7 results and the current v6 manuscript remain unchanged. This new
experiment tests a compact **vision transformer**, not another ResNet depth.
Detailed prospective protocol: `protocol/stage8/STAGE8_PROTOCOL_v1.md`.

## Run development first in Windows CMD

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage8A_vit_development.cmd
```

The nine mechanics tests and artificial CPU workflow rehearsal already passed.
This command starts the **real GPU development**, not the six evaluation seeds:

- Four full-AdamW baseline runs: two learning rates on two development seeds.
- Two LGPB calibration runs at the selected rate, if baseline quality passes.
- All six use 150 epochs, a fixed 45,000/5,000 training/validation split and
  native LayerNorm. Development never loads CIFAR-10's test split.

The small transformer is randomly initialized with six attention blocks, width
192 and 4 x 4 patches. The prefix includes tokenization/position parameters and
the first two blocks. AdamW, weight decay 0.05, label smoothing 0.1, dropout 0.1,
five-epoch LR warm-up and cosine decay are fixed in the protocol. No pretrained
weights or new dependencies are downloaded. CIFAR-10 is already available.

Baseline selection uses mean final validation accuracy over learning rates
0.0003/0.001 on seeds 1001/1002. The chosen baseline must average at least 85%,
with neither seed below 83%; otherwise the launcher stops and preserves its
screen for review. These are prospective development-readiness thresholds,
not claims of statistical significance or publication quality. A failure does
not authorize silently changing the recipe or opening evaluation seeds.

**Send `results/stage8A_vit_development/` after this stage**, including
`DEVELOPMENT_REPORT.md`, `baseline_validation_screen.csv`,
`baseline_selection.json`, and the per-run summaries/traces. If it stops at the
quality guard, send the same folder and last message. We will review baseline
credibility and implementation before the evaluation. No numeric ViT result
has been assumed in the manuscript.

Stop other substantial GPU jobs before starting. A recent preparation-time
snapshot showed the RTX 5080 heavily utilized. The launcher checks current
utilization and refuses a busy GPU; retry the same command after other work
finishes. No clocks or power limits are changed. Telemetry during runs is saved.

## Later frozen evaluation

After development review:

```cmd
call scripts\run_stage8B_freeze_and_evaluate.cmd
```

It writes `protocol/stage8/FROZEN_EVALUATION.json`, binding the chosen LR,
development-derived periodic profile, settings, source/protocol hashes and
reserved seeds **1101–1106**, then runs **24 trajectories** on the full
CIFAR-10 training/test splits:

1. Full-gradient AdamW.
2. LGPB with unchanged rho 0.95 and forced-full rule.
3. Periodic partial updates matched to independent development rates.
4. Previous-displacement replay, including its true overhead.

Order/reversal pairs balance every pairwise ordering 3/3 across seeds. This is
an architecture extension under AdamW; it is not a new Adan controller or proof
of a unique gate/state advantage. New primary criteria are prospectively recorded
in the Stage 8 protocol; old Stage 6 inferences are not changed. All mechanisms,
targets, failures and confidence intervals must be reported.

Outputs: `results/stage8B_vit_evaluation/`, including paired estimates,
`STAGE8_REPORT.md`, all target crossings and `stage8_integrity_audit.json`.
Each trajectory has `run_config.json`, `epoch_trace.csv`, `step_trace.csv`,
`gpu_telemetry.csv`, `final_checkpoint.pt` and atomic `summary.json` completion
marker. Full model and optimizer states are retained; allow roughly 2 GB for
new checkpoints/traces. The two stages are substantial GPU work, not smoke tests.

The same launcher skips matching completed runs and restarts an incomplete
variant from initialization; it does not resume a mid-epoch checkpoint.
Changed code/settings/environment or a modified freeze fail explicitly. Keep
the old outputs and use a prospectively documented new directory/version for
any necessary amendment; do not overwrite or selectively rerun unfavorable seeds.

## Diagnostic commands

Reproduce the artificial CPU checks, without CIFAR training:

```cmd
call scripts\run_stage8_00_sanity.cmd
```

Recompute completed evaluation summaries without training:

```cmd
"%PYTHON%" -B scripts\stage8_workflow.py summarize
"%PYTHON%" -B scripts\stage8_audit.py --root results\stage8B_vit_evaluation --freeze-file protocol\stage8\FROZEN_EVALUATION.json
```

Validation record: `protocol/stage8/VALIDATION.md`. Full public code/data
archiving and the remaining editorial/supplement changes are still publication
tasks. They are not replaced by adding an architecture or claiming future success.
