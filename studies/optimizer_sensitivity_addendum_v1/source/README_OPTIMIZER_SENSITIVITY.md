# Optimizer sensitivity diagnostic — 2026-10-27

Purpose: measure how FP32/BF16 gradient differences translate into AdamW
updates when parameters and optimizer history are exactly matched.

Six short cases use the two previous initialized seeds and two archived trained
checkpoints. Initialized models use synthetic inputs; trained checkpoints use
synthetic inputs and one fixed training-only CIFAR10 batch. At each of two shared
FP32 trajectory anchors, compare native/reference attention under FP32/math,
BF16/math and BF16/automatic dispatch. Contrast fresh versus retained optimizer
history at the same weights. Dropout is disabled to isolate arithmetic.

The archived history uses the common diagnostic learning rate 1e-4; its original
source rate is recorded. This is a controlled diagnostic, not an exact resumption
of source training. No rate/epsilon tuning, full training, timing comparison,
validation/test evaluation or retrospective change to the earlier failed gate.

Preparation is validated and frozen. Run in CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_optimizer_sensitivity.cmd
```

Keep other GPU work stopped. Allow several GB for raw gradients, anchor states
and measured updates. The full design contains 12 anchors, 90 forward/backward
calls, 132 one-step replays and 12 reference trajectory updates. Replays use
saved gradients and do not add forward/backward calls. Completed work is
hash-verified and skipped; retain all attempt directories if interrupted.

Return `results\optimizer_sensitivity_v1` including subfolders. The audit
recalculates statistics from raw saved tensors without repeating model gradients.
An audit PASS does not mean BF16 accuracy equivalence or a performance benefit.
Detailed design: `protocol/optimizer_sensitivity/DIAGNOSTIC_PROTOCOL_v1.md`.

Current preparation: `results/optimizer_sensitivity_preparation_v1/REPORT.md`.
Organized code: `paper_support/optimizer_sensitivity_addendum_v1/`.

Four CPU and four CUDA tests passed. Tiny CPU/CUDA workflows and raw-tensor
reconstruction passed; completed-run skipping was checked. Both real checkpoint
parameter/optimizer mappings and the training-only batch were validated without
full-model forward/backward. The full diagnostic has not run.
