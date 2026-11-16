# Idea2: pretrained adaptation with selective updates

Stage12A is prepared as a bounded opportunity screen: ImageNet-pretrained
ResNet-18 transferred to CIFAR-10, two fresh development seeds, five arms and
12epochs each. The official checkpoint is already downloaded and verified.
No real Stage12A trajectory was executed during preparation.

The comparisons are full fine-tuning, head-only training, head warmup then full
fine-tuning, head warmup then fixed layer4/head training, and the same suffix
training with a full update every fourth step. All head-warmup paths share the
same first two epochs. BatchNorm statistics are fixed in every arm; skipped
parameters and their Adam states truly pause. This is distinct from prior
zero-gradient moment continuation.

Run in Windows CMD after stopping other GPU workloads:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage12A_adaptation.cmd
```

The launcher runs artificial CPU/GPU checks before10real runs /42,240updates.
Reserve roughly3GB for checkpoints and traces, more for incomplete attempts.
Output: results/stage12A_adaptation_v1. When it completes, say “done”.

Rerunning skips completed arms; incomplete arms start a new numbered attempt.
Completed-study rerun performs read-only verification. Do not edit frozen files.
If a run errors, retain the output and send the traceback before changing knobs.

The fixed budget uses96px input, not the official224px inference transform.
It tests marginal adaptation cost given common external pretraining. It cannot
establish the best possible full fine-tuning baseline or a new method. At least
one full-backbone baseline must sustain90% in each seed; otherwise readiness
fails and the efficacy screen is inconclusive. No automatic extension.

Periodic prefix updates must retain final accuracy within0.30percentage points
of the best ordinary control and save at least10% complete-run time against both
full and the fastest quality-qualified control, in BOTH seeds. A competitive
head-only or fixed-suffix method therefore blocks an unnecessary refresh policy.

The growth pilot is retained as potentially useful mixed evidence:22–24% time
saving, improved accuracy over small controls, and inconsistent final recovery.
It failed its frozen both-seed criterion; this is unchanged. All earlier raw
results and manuscript files remain intact.

Protocol: protocol/stage12/ADAPTATION_SCREEN_v1.md.
Validation: protocol/stage12/VALIDATION_v1.md.
Current project master: adaptive_gradient_refresh_project_master_v7.4.md.
