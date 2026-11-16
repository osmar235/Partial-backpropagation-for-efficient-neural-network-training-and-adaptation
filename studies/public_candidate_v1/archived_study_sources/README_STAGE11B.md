# Stage11B: test whether growth produces useful training savings

Stage11A passed: 0 changed predictions across10,000 validation checks, narrow
steps35.22% /44.78% cheaper, expanded steps essentially the ordinary large-model
cost. Review: results/stage11A_review_v1/REPORT.md. This is feasibility evidence;
we still need to establish comparable final accuracy at lower complete-run time.

## Run from Windows CMD

Stop other GPU work, then paste:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage11B_growth_pilot.cmd
```

The launcher runs artificial CPU/CUDA checks and then ten real training runs:
two fresh development seeds, five arms,100epochs each. Arms are full, small,
full with Adam reset, small with Adam reset, and small-to-large with Adam reset.
The sole transition is after epoch50. No learning-rate restart. Expect a longer
run than the short feasibility diagnostic; reserve roughly1GB for records and
checkpoints, plus space for any incomplete attempts. No fixed runtime promised.

Output: results/stage11B_growth_pilot_v1. On successful completion, say “done”.
Rerunning the same command skips completed arms; an incomplete arm starts a new
attempt. It does not resume halfway through that arm. A completed study is
re-audited without changing its output files. Do not edit frozen source files.

Success requires both seeds: final growth accuracy within0.2percentage points
of the best ordinary control, at least5% less complete-run time than full and
the fastest qualifying control, and full/growth sustained80% readiness. All
training, growth/reset, validation, warmup and checkpoint time is charged within
the declared clock. Dataset loading and post-run audit/sealing are outside it.
All outcomes are development evidence; publication remains on hold.

Protocol: protocol/stage11/GROWTH_PILOT_v1.md.
Freeze: protocol/stage11/FROZEN_GROWTH_PILOT_v1.json.
Project master: adaptive_gradient_refresh_project_master_v7.2.md.
