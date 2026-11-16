# Stage14A: intermediate token compression

Prepared 2026-10-23 before real results. Author approved idea 4. This is a
fixed-checkpoint development diagnostic, with no optimizer updates or new
training. It asks whether intermediate compression has useful accuracy/cost
headroom on our existing small ViT. A pass authorizes review, not an automatic
training pilot or a publication claim.

## Motivation and related work

Stage10 reduced spatial detail at the input and lost substantial accuracy.
Here all 64 patches pass through the first three transformer blocks, then
compression reduces subsequent token processing. Learned features may tolerate
merging better than raw pixels; this is a hypothesis, not an established result.

[Token Merging](https://arxiv.org/abs/2210.09461) combines similar ViT tokens
and studies inference and training efficiency. Our similarity arm is inspired
by its bipartite matching and proportional attention, but uses residual token
features after one complete block, not its attention-key metric and gradual
schedule. It is not a ToMe reproduction or a novel generic merging method.
[DynamicViT](https://arxiv.org/abs/2106.02034) is related learned token pruning;
we do not implement its learned scoring or training procedure.

## Fixed design

Use Stage11B full final checkpoints, seeds 1501/1502, epoch 100, 17,600 updates.
Reuse the existing 45,000/5,000 training/development split. No CIFAR10 test data.
These seeds and validation examples have been examined previously; conclusions
are exploratory and require a fresh prospective confirmation if promising.
Model: 32px images, patch4, width192, six blocks, three heads, 64 patches + CLS.

Three arms share every weight and use the original classifier:

| Arm | Operation after block 3 |
|---|---|
| full | Native model, all 65 tokens through all six blocks |
| spatial | Average adjacent horizontal patch pairs: 64 to 32 patches |
| similarity | Each even raster patch joins its most cosine-similar odd patch |

Both compressed arms retain CLS unchanged. Similarity uses normalized residual
features, deterministic first-index argmax ties, detached routing, and
differentiable means. All 32 destinations remain, even if they attract no
sources. All input patches contribute; no gradient detachment of features.
Group sizes sum to 64; CLS size is 1. Every remaining attention block adds
log(group size) to each key's attention logits. This preserves multiplicity
when identical tokens are merged and avoids overweighting the unmerged CLS.
Positions are embedded before the first block; no new positions are added.
No learned merger parameters, reconstruction, auxiliary heads or optimizers.
Dense one-hot matrix multiplication implements deterministic group reduction;
its allocation and cost are included. It may be inefficient on this small model.

## Quality and mechanics

Score all 5,000 development examples once per arm/state. Save per-example
logits and labels for independent recomputation of cross entropy, accuracy,
prediction disagreements and logit RMS changes relative to full. Full must
reproduce the frozen source accuracy exactly (82.46%, 82.74%) and loss within
1e-5; otherwise abort before interpreting compression. Source accuracy must
also be at least 80%. Weights and buffers must remain byte-identical throughout.

Artificial tests check uncompressed logits/gradients, merged mass and gradients
to every token, proportional-attention equivalence for duplicated tokens,
native architecture finite gradients, and decision-rule boundaries. Neither
similarity routing nor compression is assumed function-preserving in general.

## Resource measurements

Measure inference (eval/no_grad forward plus scalar readback) and forward+CE+
backward (train mode, no optimizer step). CE uses training labels with no MixUp,
augmentation or label smoothing: an operation microbenchmark, not a replay of
the original training recipe. Backward must produce finite gradients for all
parameters, including the patch embedding. Same training batches and dropout
seed per trial across arms; changed sequence shapes prevent identical masks.

Twelve distinct shuffled training batches per state, batch256; three warmups
per arm/operation. Rotate arm order by trial and seed, and alternate operation
order across trials. Log every timing, order, loss/readback, gradient check and
CUDA allocation peak. Total 144 measured operations plus 36 warmups. No parameter
updates, calibration, LR choices or reset-optimizer utility comparison.

Data are device-ready before timing. Synchronize CUDA before/after every
operation. Include merging, mask construction, all forwards/backwards, gradient
zeroing and scalar readback; exclude common normalization/data loading,
validation, telemetry and post-timing gradient audits. Gradients are also
cleared before resetting peak counters so each operation starts without them.
Report absolute peak allocated/reserved memory and allocated increase above
that operation's baseline. Resident data/model storage is visible in absolute
peaks. There are no optimizer buffers; these are NOT full optimizer-step or
end-to-end training costs. Warmup and data/validation overhead are recorded
separately. Strict FP32, math attention, deterministic execution, no compilation.

## Prospective engineering filters

For each compressed arm, BOTH source states must retain accuracy within 0.30
percentage points of full and achieve either at least 10% lower mean
forward/backward time OR at least 20% lower incremental allocated peak memory.
Memory ratios use mean per-operation allocated increases; never reserved peaks.
Source reproduction and >=80% readiness are prerequisites. Apply the same rule
to spatial averaging; similarity is primary, spatial is a practical control.
If spatial passes too, similarity needs a demonstrated advantage over it before
claiming value for content-aware matching. No winner is selected automatically.
Inference time/memory, prediction changes and loss shifts are descriptive.
No p-values or seed-population inference from two previously used checkpoints.

A failure parks this specific 50%-at-midpoint, no-adaptation configuration;
it does not exclude token compression, other boundaries/rates, or adaptation.
A pass motivates a separately frozen training/confirmation study; it does not
demonstrate learning speed, energy savings, or time to target. No tuning sweep.

## Provenance and reruns

Freeze spec, implementation, tests, launcher, runtime ViT source, evidence,
source checkpoint bytes and split hashes before real execution. Preserve old
studies. Complete outputs are hash verified and audited read-only on rerun.
An interrupted attempt is retained; a new attempt starts from original states.
Raw completion binds all attempt files; final completion additionally binds the
independent arithmetic audit and summary. Use a new protocol version for changes.
Artificial CPU/CUDA runs exercise the same path on tiny random networks/data;
their quality/resource numbers have no scientific interpretation.
