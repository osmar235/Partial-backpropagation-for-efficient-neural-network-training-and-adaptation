# Stage10A coarse/detail representation screen — 2026-10-21

Author-approved new hypothesis after Stage9E failure: can structured coarse/detail
inputs preserve useful learning while doing expensive processing at lower spatial
resolution? This is a representation/architecture screen, not a repair of the
closed norm/loss gate. Standard full AdamW and one joint classification loss in
every arm; no partial gradients, local independent losses, random optimizer,
evolution or claimed new method yet.

## Five fixed arms

1. full_rgb32: ordinary native ViT6, width192, MLP768, heads3, patch4, input32.
2. small_rgb32: ordinary input32 ViT6 width96, MLP384, heads3, patch4.
3. coarse_rgb16: avg-pool2 input, input16 ViT6 width192, MLP768, patch4.
4. pixel_branches: same input16 ViT body on top-left pixel phases; tiny detail
   CNN on the other three phases. Lossless pixel grouping, no wavelet bands.
5. haar_branches: same body on average/coarse image, same tiny CNN on three
   orthonormal Haar detail bands. LL divided by2 equals ordinary avg-pool2.

The dual branches share the same trainable shapes, parameter count, component
initialization and fusion rule: concatenate class feature with48-channel detail
feature after spatial mean/LayerNorm, then linear classifier. Detail CNN uses
9->24->48 channels, two conv3 stride2 padding1 with GELU. No auxiliary loss or
detached/frozen branch. Both branches train from the same joint loss.

For each2x2 block a,b,c,d: LL=(a+b+c+d)/2; H=(a-b+c-d)/2;
V=(a+b-c-d)/2; D=(a-b-c+d)/2. All bands are retained; transform is invertible
and energy preserving. Expensive ViT sees16x16; the9-channel16x16 detail is
handled by the small CNN. Fewer spatial tokens do not mean fewer source
coefficients. The pixel control is also lossless, with identical dual-branch
capacity. Neither branch split makes optimization mathematically independent.

Haar-vs-coarse alone confounds extra capacity and detail. Pixel grouping controls
trainable capacity/graph shape, but different conditioning and branch allocation
remain parts of the representation effect. This does not isolate a universal
frequency mechanism or prove that Fourier/wavelets are new.

## Fixed training and randomness

Two new DEVELOPMENT seeds1401/1402, five arms each,100epochs. Reverse run orders
across seeds; not fully balanced positions. Existing training-only45k/5k split,
already adaptively reused; CIFAR10 test dataset is never constructed.
Same crop/flip/normalization/MixUp streams and example order per seed; transforms
are applied INSIDE the model AFTER common augmentation and mixing. Evaluation
transforms also count in time. All parameters updated every step.

Same fixed recipe as the prior baseline: AdamW LR.001/wd.05, MixUp alpha.2,
label smoothing.1, batch256; first100epochs of300epoch cosine horizon, five
LR-warmup epochs. No architecture-specific LR tuning or pretrained weights.
Negative results apply to this bounded recipe, not optimally tuned architectures.

Component-specific CPU seeds make input16 bodies identical initially across
coarse/pixel/Haar arms and make the two detail branches identical initially.
Classifier heads start at zero. Model dimensions differ across full/small/
low-resolution arms, so initial tensors and dropout masks cannot all be equated.
Reset torch seed per minibatch to seed+195001+step BEFORE forward, inside timing.
Compatible input16 bodies therefore use matching dropout draws; differing-shape
models are only reproducibly seeded, not identical-mask counterfactuals.

Enforce CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization, one CPU
thread, deterministic algorithms (errors not warnings), cuDNN deterministic,
no TF32/AMP/compile, and math-only attention. Verify repeated same-model training
mechanics on CPU and CUDA before freezing. These settings can affect throughput;
all new arms use them and old nondeterministic timings are not direct baselines.
No silent fallback if an unsupported deterministic operation is encountered.

## Timing and fixed development decisions

Exclude16 disposable warmup updates, then restore model/Adam/data RNG state.
Actual augmentation, transform, full backprop, optimizer, seed resets and step
logging preparation are in active-loop time. Primary observed time starts after
setup/warmup and includes training, validation, trace writes, prints and scheduled
telemetry, stopping when THREE consecutive80% validations are confirmed. Final
checkpoint writing excluded from this target clock; report separate experiment,
warmup, validation, training and checkpoint clocks. Evaluate every epoch.

Primary practical flag for Haar, required on BOTH seeds:
- Reach sustained80%, as must full_rgb32 (baseline readiness).
- >=5% shorter observed time than full AND fastest reaching ordinary baseline
  among full_rgb32/small_rgb32/coarse_rgb16.
- Final accuracy within0.2pp of the highest final accuracy among those ordinary
  baselines. Retain all target censoring; no post hoc threshold changes.

Separate necessary signals for a promising structured-detail hypothesis:
- Endpoint detail gain >=0.5pp over coarse_rgb16 on BOTH seeds. This fixed-epoch
  comparison alone is not a compute-matched benefit; primary timing handles cost.
- Endpoint structure gain >=0.2pp over capacity-matched pixel_branches AND
  active-training wall ratio <=1.05 to pixel_branches, BOTH seeds.
All three flags must pass to mark this development hypothesis promising. These
are chosen screening thresholds, not significance tests or a publication result.
Secondary82% attainment and curves are descriptive, not alternative pass criteria.
If full baseline misses80% in100epochs, report lack of readiness and stop; do not
lower targets, extend epochs or tune architectures automatically.

Keep all100epoch results; no best-epoch substitution. Two seeds cannot establish
uncertainty or broad superiority. Strict execution and repeat tests improve
reproducibility on this environment, not across all hardware/releases.

## Freeze, interruption and interpretation

Freeze code/spec/protocol/tests/launcher and prior evidence before scientific
execution. Matching completed trajectories verify hashes and skip. Interrupted
trajectories restart from initialization in NEW attempts; retain incomplete
traces. No mid-run source/config changes or silent state reuse. Audit both data
streams, transform identities, all-step optimizer counters, checkpoints, targets
and practical/structure flags before completion. CPU/CUDA synthetic outputs are
separate and scientifically meaningless. No automatic follow-up training.

Prior science and failed Stage9E remain unchanged. Publication stays on hold.
Prior art includes [spectral representations](https://arxiv.org/abs/1506.03767)
and [wavelet CNNs](https://arxiv.org/abs/1805.08620). Their existence motivates
explicit controls and prevents treating a wavelet layer as a novelty claim.
[PyTorch2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
distinguishes deterministic-operation settings from seeding alone.
