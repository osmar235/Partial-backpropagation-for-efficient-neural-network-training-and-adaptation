# Stage12A: pretrained adaptation and selective updates

Prepared2026-10-22 before real transfer results. The author approved moving to
idea2 after the growth pilot. Preserve growth as a potentially useful observed
accuracy/time tradeoff with inconsistent quality recovery; do not rewrite its
failed both-seed rule or claim novelty. This stage tests a distinct context:
can selective updates exploit already-learned representations?

## Scope and prior work

Use the official torchvision ResNet18 ImageNet-1K V1 weights, with a new seeded
ten-class head. Exact checkpoint SHA256 and metadata are bound to the freeze.
No prior project checkpoint serves as a free pretrained initialization.

- Official model and normalization: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Adaptive freezing is prior art: https://arxiv.org/abs/2102.01386
- Head-first then full fine-tuning is prior art: https://arxiv.org/abs/2202.10054

These sources motivate controls, not claims that their published results transfer
to this setting. This is a conventional-baseline opportunity screen, not a new
algorithm, AutoFreeze reproduction, external benchmark or publication result.
Its purpose is to determine whether there is useful room for a later mechanism.

## Fixed design

Fresh development seeds1601/1602; five arms;12epochs each;352batches/epoch;
4,224updates/run;42,240 total updates. Reverse order for the second seed.
Original CIFAR-10 training-only45,000/5,000 split. Reused validation remains
development evidence; never load train=False. Identical pretrained backbone and
seeded head across arms. No class filtering or selection based on results.

| Arm | First2epochs | Remaining10epochs |
|---|---|---|
| full | All weights | All weights |
| head | Head only | Head only |
| lp_full | Head only | All weights |
| lp_suffix | Head only | layer4 + head |
| lp_periodic | Head only | layer4 + head; full every4th update |

The periodic index resets once at the start of epoch3: its first adaptation
update is full, then three suffix updates. It runs across epoch boundaries.
Prefix comprises stem and layer1–3. No gradients/backward graph for the frozen
prefix on suffix steps; no backbone graph on head-only steps. On frozen steps,
parameters have grad=None: Adam moments, step counters, weights and decoupled
weight decay pause. This differs from the earlier zero-gradient moment
continuation method and must not be described as its direct replication.

BatchNorm RUNNING STATISTICS remain at pretrained values in ALL arms, including
full; model.eval() does not disable autograd. BatchNorm affine weights follow
their active parameter group. This prevents hidden feature adaptation through
buffers and makes frozen-path semantics explicit. Standard BN-updating
fine-tuning remains an untested alternative, limiting conclusions about the
best achievable full baseline. No feature caching: each training sample is
newly augmented and forward cost is charged on every step.

All arms share data order/crop/flip streams. The head/lp_* arms must match
bitwise through epoch2: weights, moments, losses and validation histories.
No optimizer reset at unfreezing; newly active parameters begin empty while
head Adam continues. Global LR schedule does not restart.

## Budget recipe and readiness

Native32px CIFAR images receive reflection-pad4 random crop32 and horizontal
flip, then bilinear antialiased resize96 and ImageNet mean/std normalization.
Validation resizes the whole32px image to96 without augmentation. This is a
budget96px screen, deliberately DIFFERENT from the official224px inference
recipe. It does not test the strongest possible pretrained baseline.

Batch128, eval256; AdamW backbone lr1e-4/head lr1e-3, wd.01; no MixUp or label
smoothing. One-epoch LR warmup then cosine over12epochs to0.01 times base LR.
Head-only start is a fixed two-epoch warmup, not an optimized linear probe.
All knobs and the one prefix boundary are fixed before outcomes. No search.
Strict deterministic FP32 execution inherited from previous controlled screens;
no AMP, TF32 or optimized-BN comparison. Both sides share these restrictions.

At least one full-backbone baseline (full or lp_full) must sustain90% validation
for two consecutive epochs in EACH seed. Otherwise the opportunity screen is
inconclusive due to baseline readiness. Finish the fixed12epochs and retain
all arms; do not extend, change LR, resize or lower readiness after seeing data.

## Prospective opportunity rule

Primary candidate: lp_periodic. Ordinary controls: full, head, lp_full, lp_suffix.
For each seed, identify best final epoch12 control accuracy. Quality-qualified
controls are within0.30percentage points of that best. Candidate must also be
within0.30points, and its complete-run wall time must be at least10% lower than
BOTH full and the fastest quality-qualified control. Candidate must sustain90%
for two epochs and the full-baseline readiness condition must hold. Require all
conditions in BOTH seeds. A fast competitive head-only or fixed-suffix control
blocks an unnecessary periodic policy. All controls remain visible regardless.

Report candidate's accuracy difference to fixed suffix to assess whether prefix
refresh adds anything; this descriptive contrast has no post hoc minimum gate.
Report90%/92% crossings and censoring at12. All endpoint results use final epoch,
not best epoch. A passed screen supports further investigation, not statistical
significance, novelty or publication. A failed screen rejects this recipe only.

## Costs, state and preservation

Primary per-run clock begins before constructing/loading the model and ends
after final checkpoint save and CUDA sync. Charge initialization, six discarded
full hardware-warmup updates, weight/buffer restoration, fresh Adam, all actual
training including the two head-warmup epochs, validation every epoch,
warmup/final checkpoints, trace writes, telemetry and console work.
No precomputed features and no free trained warmup are shared across arms.

Dataset loading, official checkpoint acquisition, input hash verification,
post-run summary/sealing and the independent audit are outside per-run clocks;
data loading is logged. ImageNet pretraining is a common external sunk cost:
results concern marginal adaptation ONLY, never end-to-end learning cost.
Initialization/checkpoint IO can affect small-run timings; compare training
seconds separately and retain full clocks for the frozen rule. Reserve3GB.

Freeze source/spec/tests/launcher/protocol, official checkpoint full hash and
supporting evidence before real output exists. Record per-step mode/counters/LR/
loss, per-epoch metrics/times, exact epoch2 and final states/moments/data RNGs.
Audit all group ages, frozen weights and BN buffers, matched warmups, data
streams, schedules and decisions. CPU/CUDA artificial workflows and native
synthetic workload tests precede the freeze; they establish no scientific result.
Completed arms are immutable; retry starts a fresh attempt for incomplete arms.
Completed-study rerun re-audits without writes. Review before further work.
