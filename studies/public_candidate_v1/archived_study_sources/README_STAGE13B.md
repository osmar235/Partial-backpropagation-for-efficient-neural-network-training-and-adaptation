# Fix the reference before interpreting local learning

Stage13B takes priority over the prepared Stage14 token experiment. It restores
the same two Stage12 head-warmup checkpoints and runs one fixed epoch per arm:
original optimizer continuation versus reset Adam state, with identical batches,
augmentations and scheduled learning rates. Four short trajectories,1,408 updates.

The continuation must reproduce all352 original losses per seed and the known
epoch3 validation results. Fixed intermediate evaluations show whether early
loss deterioration resolves. The reset arm isolates classifier optimizer history;
no local method, LR search or automatic follow-up is included.

Run in CMD with other GPU work stopped:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage13B_reference.cmd
```

Output: `results/stage13B_reference_v1`. Say "done" when complete. A completed
rerun verifies output read-only. Interrupted attempts remain preserved.
Integrity PASS does not imply that reference reproduction or qualification
passed; review `audit.json`, curves and comparisons before doing anything else.

This checks an already successful, previously seen development recipe. It is
not fresh confirmation, a local-learning result or a resource benchmark.
See `protocol/stage13b/REFERENCE_REPAIR_v1.md`. Stage13A remains unchanged and
inconclusive for utility; Stage14 remains prepared but queued. Publication stays
on hold. A matched local-learning trajectory requires its own subsequent protocol.
