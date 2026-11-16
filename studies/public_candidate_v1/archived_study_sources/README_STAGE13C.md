# Stage13C: local learning over a matched training trajectory

Compare full, connected auxiliary, detached local and suffix-only training.
Two existing source states;352updates per arm. Restore the classifier's Adam
history in every arm, reproduce auxiliary calibration and restore its history
for local/full_aux. Use identical original batches, augmentation and scheduled
rates. The full reference must reproduce Stage12/13B. No first-step rejection.

Run in CMD, with other GPU work stopped:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage13C_local_trajectory.cmd
```

Output: `results/stage13C_local_trajectory_v1`. Say "done" when complete.
Eight trajectories total2,816updates, plus256head-calibration steps.
Completed reruns verify files read-only; interrupted attempts remain preserved.

Judge the fixed endpoint, with intermediate recovery curves retained. Local
must improve and remain within0.30pp of the best control in both seeds, with
>=10% charged training-time savings OR >=20% incremental operation-memory savings
against the specified quality-qualified controls and full. Charge all auxiliary
calibration cost to each arm that uses it. A memory-only pass is not a speedup;
calibration memory is reported separately and must be considered in review.

See `protocol/stage13c/LOCAL_TRAJECTORY_v1.md` for exact criteria and accounting.
This is a reused-development-state screen, not new confirmation or a publication
claim. No tuning or automatic extra epochs. Prior studies remain unchanged;
Stage14 stays queued and publication remains on hold.
