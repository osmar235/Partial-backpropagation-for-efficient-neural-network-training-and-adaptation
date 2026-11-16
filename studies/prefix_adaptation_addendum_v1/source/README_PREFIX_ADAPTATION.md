# Run the prefix adaptation quality/cost study

Prepared 2026-10-28. Full scientific run pending. Close other GPU workloads.
Use Windows CMD and the existing project environment:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_prefix_adaptation.cmd
```

Return `results\prefix_adaptation_v1` when COMPLETE appears. The launcher audits
all checkpoints automatically, including final inference in both scoring modes.
The full run has 36 five-epoch trajectories: three arms, two trained sources,
six counterbalanced timing sessions. No package installation is needed.

All arms use BF16/automatic forward and explicit FP32 loss. Compare full updates,
a permanently frozen prefix through block 3, and prefix refresh every four
updates. The remaining three blocks, final norm and classifier update each step.
Original source weights and common fresh AdamW; no cost-screen endpoint or
calibration state is reused. Measure complete adaptation time and final accuracy
against the fixed 0.30-percentage-point margin in both evaluation modes.

A completed rerun audits without training; its checkpoint inference is expected.
An interrupted run preserves completed trajectories, resumes missing ones and
flags the session as fragmented, preventing a clean timing qualification. Send
such results for review; do not manually delete attempts or change the freeze.

The frozen-prefix control asks whether refreshing adds useful quality. Earlier
Stage9 failures remain unchanged. This is a bounded exploratory adaptation
study, not a novelty claim or permission for a boundary/refresh sweep.

Protocol: `protocol/prefix_adaptation/ADAPTATION_PROTOCOL_v1.md`.
