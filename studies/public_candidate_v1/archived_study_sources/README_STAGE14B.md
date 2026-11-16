# Stage14B — bounded token accuracy recovery

Five fixed epochs per arm and seed, six runs, 5,280 recovery updates. The
three arms are full, spatial averaging and similarity merging. The compressed
arms keep the same 65 -> 33 token reduction after block 3. Source AdamW history
is restored in every arm; calibration/setup/warmup costs are not hidden.

From Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage14B_token_recovery.cmd
```

Stop other GPU work first. The launcher runs artificial CPU/CUDA checks, then
the real fixed study and its audit. Let all six runs finish even if intermediate
accuracy drops. Only the final fifth epoch decides; no automatic extensions.

Output: `results\stage14B_token_recovery_v1`. Send that folder for review.
Rerun the same command after interruption: completed arms are checked/skipped,
and an incomplete arm restarts from its source in a new retained attempt.
Changed frozen files or environment block continuation rather than overwrite.

Both seeds must meet the 0.30-point quality limit against the original full
source and adapted controls, plus the frozen recovery resource criterion.
The report separately decides whether similarity warrants priority over spatial.
A pass is development feasibility, not a new-method or from-scratch speedup
claim. Original source training costs remain in the summaries. No test scoring.

Protocol: `protocol/stage14b/TOKEN_RECOVERY_v1.md`.
Preparation checks: `protocol/stage14b/VALIDATION_v1.md`.
