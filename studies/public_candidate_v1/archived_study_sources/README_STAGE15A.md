# Stage15A: side-network quality and complete adaptation cost

Ten runs: five arms × original source seeds 1501/1502, five epochs each.
Compare full, head-only, head-then-full, final-feature side, multi-depth side.
The last is the candidate; all four others are required controls. Existing
checkpoints and previous results remain untouched. No real run occurs in setup.

From Windows CMD, with other GPU work stopped:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage15A_side_adaptation.cmd
```

Wait for `COMPLETE`. Send `results\stage15A_side_adaptation_v1` for review.
If interrupted, rerun the same command. Completed arms are verified and skipped;
partial attempts are preserved and restarted. Mixed-launch timings require review.
Do not change frozen files or delete partial results to force a rerun.

Success requires final accuracy within 0.30 percentage points of the strongest
control/source and at least 10% lower complete adaptation time versus full and
the fastest qualified control, in both seeds. This can fail even if side updates
are cheaper. The full reference must qualify and reproduce the prior trajectory.
The test set remains unused. No automatic follow-up, significance or novelty claim.

Protocol: `protocol/stage15a/SIDE_ADAPTATION_v1.md`; specification and environment
freeze are in the same directory. Preparation validation is recorded there.
