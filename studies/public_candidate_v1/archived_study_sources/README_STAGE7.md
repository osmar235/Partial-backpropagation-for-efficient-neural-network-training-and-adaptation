# Stage 7 reviewer follow-up — local v1.1

Updated 2026-10-18. New post-confirmation experiments; original Stage 6 code,
results, preregistration and release are preserved. Current decision log:
`adaptive_gradient_refresh_project_master_v6.4.md`. Detailed prospective plan:
`protocol/stage7/REVIEW_STAGE7_PROTOCOL_v1.1.md`.

The supplied ZIP assumed a different repository layout. The installed scripts
use this working project's source files, existing CIFAR-10 data and archived
seed folders. You do not need to extract or merge another ZIP.

## Current status: all follow-up experiments complete

Stage7 and the separate second-architecture Stage8 study are complete. See
`README_STAGE8_RESULTS.md` for the final review and manuscript v7. All planned
training is complete; earlier next-run statements below are historical and
do not change any original result or prospective record.

Stage 7E completed all 12 AdamW runs and passed independent audit. See
`results/stage7E_review_v1/REPORT.md`. LGPB saved 9.66% training wall time,
accuracy delta +0.028 pp (exploratory 95% CI -0.106 to +0.163). This supports
recipe robustness without proving accuracy equivalence or superiority.

Stage 7C completed all 36 sensitivity runs and passed independent trace audit.
See `results/stage7C_review_v1/REPORT.md`. Removing forced-full refreshes or
skipping a larger prefix saved additional time but lowered accuracy. EMA
alternatives showed no clear joint improvement. Timing varied across and within
runs, limiting fine speed comparisons. Keep the frozen controller unchanged.

The full Stage 7A–7E plan is complete (96 new training runs).

Stage 7D completed all 12 CIFAR-100 runs and passed independent audit. See
`results/stage7D_review_v1/REPORT.md`. LGPB saved 9.98% training wall time, with
mean accuracy loss 0.695 pp (exploratory 95% CI -1.358 to -0.032).
This demonstrates a speed–accuracy trade-off, not accuracy preservation.

Stage 7B completed all 36 runs and passed independent trace validation. See
`results/stage7B_review_v1/REPORT.md`. LGPB saved 8.52% wall time versus Adam,
with mean accuracy difference -0.108 pp. It beat prefix freezing on accuracy;
matched random/periodic skipping and displacement replay remained competitive.
The data do not demonstrate loss-gate superiority. No experimental settings
were changed after reviewing these outcomes.

**Manuscript revision is complete.** The current working Word/PDF manuscript is
`NeuralNetworks/LGPB_Neural_Networks_revised_v6_Stage7` with the appropriate
extension; see `NeuralNetworks/CURRENT_MANUSCRIPT.md`. It incorporates the
evidence map in `protocol/stage7/FINAL_REVIEW_SYNTHESIS.md`, including the
matched-control, CIFAR-100 and timing limitations. Next is author review and
final public archival identifiers. No additional training is required by this plan.
Earlier phase commands below are retained only for reproduction/reference;
they are not requests to rerun completed experiments.

## Run in Windows CMD

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage7B_core_ablation.cmd
```

7B runs six variants for each of seeds 51–56 (36 training runs), rotates their
execution order and generates paired/timing reports automatically. Leave other
GPU work idle while measuring. Start here and inspect its report before the
remaining phases; do not tune the frozen candidate based on new results.

Then run each phase separately, in this priority order:

```cmd
call scripts\run_stage7D_cifar100_transfer.cmd
call scripts\run_stage7C_sensitivity.cmd
```

7D: Adam versus unchanged LGPB on CIFAR-100, seeds 63–68 (12 runs). First use
downloads CIFAR-100. 7C: six sensitivity variants, seeds 57–62 (36 runs).
The predeclared settings and seed ranges do not change with execution order.

Optional AdamW robustness, seeds 69–74 (12 runs):

```cmd
call scripts\run_stage7E_adamw_optional.cmd
```

The mechanics/CPU sanity check and no-training archived audit already passed
locally. To reproduce them:

```cmd
call scripts\run_stage7_00_sanity.cmd
call scripts\run_stage7A_archived_audit.cmd
```

## Results and resume

The same launcher resumes missing/incomplete variants and skips completed ones.
An interrupted variant restarts from initialization. Do not change settings or
code in a partially completed study; configuration/hash mismatches fail explicitly.
Use a fresh output directory for a documented amended study.

Send `results/stage7B_core_ablation/` after 7B. Principal outputs:

- `stage7_summary.md`
- `stage7_all_runs.csv`
- `stage7_descriptive_paired_estimates.csv`
- `stage7_paired_vs_reference.csv`
- `stage7_step_mode_timing_summary.csv`
- `seed_*/run_config.json` and variant `summary.json`, `epoch_trace.csv`, `step_trace.csv`

Both Adam and LGPB references are included for 7B. Positive accuracy differences
favor the variant; wall ratios below one favor the variant. These exploratory
paired intervals do not reapply the original Stage 6 inferential gates.

Archived audit outputs are in `results/stage7A_original_audit/`. Supplied audit
outputs are separately preserved in `results/stage7A_supplied_reference/`.
Artificial CPU/CUDA integration outputs contain no scientific performance evidence.

See `protocol/stage7/VALIDATION.md` for the initial synthetic checks.
Stages 7B, 7D, 7C and 7E are all complete.
