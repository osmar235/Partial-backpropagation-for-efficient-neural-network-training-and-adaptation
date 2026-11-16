# Idea3: local objectives / smaller credit-assignment problems

Stage13A is a short diagnostic at two saved Stage12 head-warmup model states.
It splits ResNet-18 into three blocks with two small auxiliary prediction heads.
Four controls distinguish ordinary backpropagation, extra supervision, local
gradient cuts, and simply leaving the prefix frozen. This tests a conventional
local-learning design; it does not yet establish a new method or convergence.

Run in Windows CMD after stopping other GPU work:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage13A_local_diagnostic.cmd
```

The launcher checks CPU/GPU mechanics, then calibrates only the auxiliary heads
and performs96measured disposable single updates. It does not train a long new
trajectory. Source checkpoints remain unchanged. Reserve roughly2GB for saved
states and traces; runtime depends on the machine, but this is intended to be
much shorter than the preceding training screens.

Output: results/stage13A_local_diagnostic_v1. Say “done” when complete.
An interrupted run starts a new attempt when rerun. Completed rerun checks the
saved evidence without changing it. Do not edit frozen scientific source files.

Measurements:

- Final-loss improvement on separate training examples after each update.
- Per-block gradient agreement with ordinary full backpropagation.
- Update-operation time and GPU allocated-memory peaks.
- Auxiliary-head setup/calibration overhead and an optimistic break-even count.

The two source states are already seen. Calibration, update-probe and score pools
are disjoint within the training set; the score pool is not an unseen validation
set. No validation/test scoring is used here. Costs are device-ready operations,
not complete training time; calibration overhead is reported separately.

The fixed review filters require positive local score-loss improvement retaining
at least half of full's positive mean improvement, plus either10% faster updates
or20% smaller incremental memory peaks, in both states. These coarse engineering
flags only qualify a design for pilot review. They cannot establish final quality.
Full_aux is an important control for improvements caused by extra supervision.
No automatic pilot, transition search, LR sweep or longer calibration.

Protocol: protocol/stage13/LOCAL_DIAGNOSTIC_v1.md.
Validation: protocol/stage13/VALIDATION_v1.md.
Current master: adaptive_gradient_refresh_project_master_v7.6.md.
