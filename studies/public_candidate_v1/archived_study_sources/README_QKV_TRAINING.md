# QKV real-data adaptation confirmation

The artificial benchmark confirmed 5.2–7.1% update-time savings. This new study
tests complete real-data adaptation runtime and preserved training/validation
behavior. Eight sessions compare native attention and unchanged QKV-unbind on
two established checkpoints, five epochs each: 32 trajectories /28,160 updates.
No CIFAR-10 test-set evaluation, new approximation or other optimization.

Stop other GPU work, keep the machine awake, and run in CMD:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_qkv_training.cmd
```

Allow tens of minutes and several GB of free disk space for paired checkpoints.
Send the complete `results\qkv_training_v1` folder for review. Completed trajectories
are hash checked and skipped on replay. An interrupted session is preserved and
marked fragmented; its timings cannot establish the primary result. Do not delete
or replace unfavorable attempts. Rerun the same command to complete missing work
and return the evidence for review if interrupted.

Primary timing includes data loading, source/model setup, warmup/restore, training,
augmentation, validation and final artifacts. Imports, provenance hashing and
post-run audits are excluded. Compare against the archived full training reference
and require exact paired learning outcomes. A result below 5% is reported honestly;
there is no retroactively selected minimum useful effect.

Read-only audit (does not train):

```bat
"%PYTHON%" -B scripts\review_qkv_training.py results\qkv_training_v1
```

Protocol: `protocol/qkv_training/TRAINING_PROTOCOL_v1.md`. Original 10% screen
and all previous evidence remain unchanged. Publication still requires broader
contribution/novelty assessment, not just this timing result.
