# Stage8 v2: evaluation after completed development review

2026-10-19. The baseline passed (mean validation85.77%, minimum85.46%),
but LGPB calibration performed poorly (mean71.44%, gap-14.33pp). All four
checkpoints/traces passed integrity review, independent CPU validation
rescoring and exact suffix-gradient checks. Full review:
`results/stage8A_review_v2/REPORT.md` and
`protocol/stage8/DEVELOPMENT_REVIEW_v2.md`.

The next stage retains the already planned four-arm experiment, including
negative results. No third recipe, gate tuning, replacement seeds or changed
readiness criteria are introduced. Periodic/replay controls will help assess
the scope of this adverse result. The scientific protocol/spec/sources used
for development stay unchanged; the real freeze has been created.

## Windows CMD

Stop other GPU jobs, then run:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage8B_freeze_and_evaluate_v2.cmd
```

The launcher verifies the same freeze, then starts24 complete trajectories:
seeds1101--1106, full AdamW/LGPB/development-matched periodic/prefix replay,
300epochs each, all50k training images,10k test images, same MixUp0.2 recipe.
The settings/profile are sealed in `protocol/stage8/FROZEN_EVALUATION_v2.json`.
The periodic schedule comes from the two retained development calibrations.

Allow roughly10--12hours at development-run throughput, with substantial
uncertainty from hardware/workload conditions. Keep the GPU free. The startup
guard refuses high utilization; it does not control later competing jobs.
Telemetry and balanced/reversed execution orders are retained for drift review.

Matching completed runs skip on rerunning the same command. An unfinished
trajectory restarts from initialization. Preserve every completed result and
all telemetry, including slow/negative seeds. Changed code/settings/environment
will fail explicitly; do not edit the freeze or discard an unfavorable run.

Send **`results\stage8B_vit_evaluation_v2`** and
**`protocol\stage8\FROZEN_EVALUATION_v2.json`** when complete. The final audit
runs automatically and writes `stage8_integrity_audit.json`. Summaries include
all arms, target crossings/reach counts, paired estimates and new primary
criteria separate from Stage6. This study may establish a limitation rather
than positive transfer. The full four-arm evidence, not development success
selection, determines the final manuscript wording.
