# Stage14C — retained versus fresh optimizer history

Test whether resetting AdamW helps similarity token merging more than it helps
the matched full-token model. Twelve runs: three computations, two optimizer
conditions, two source checkpoints; five fixed epochs each (10,560 updates).

Run in Windows **Command Prompt** after stopping other GPU work:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage14C_optimizer_history.cmd
```

The launcher runs artificial CPU/CUDA checks, verifies the freeze and executes
the 12 real trajectories. Wait for COMPLETE, then send
`results\stage14C_optimizer_history_v1` for review. If interrupted, rerun the
same command: completed arms are verified/skipped, incomplete arms restart in
a new preserved attempt. Runs spanning launches require review of timing before
an efficiency claim. If a freeze/audit check fails, send the error; do not edit
the frozen files or remove existing attempts.

The endpoint quality margin stays 0.30 percentage points. Primary efficiency
requires at least 10% saving in complete instrumented adaptation cost against
full and quality-qualified ordinary controls, in both seeds. Memory is secondary.
Mechanism support is a separate paired reset interaction, not accuracy gain alone.

Protocol: [optimizer history](protocol/stage14c/OPTIMIZER_HISTORY_v1.md).
Current master after preparation: v7.17. Prior evidence and paper memory v7
remain preserved; this is exploratory, and publication remains on hold.
