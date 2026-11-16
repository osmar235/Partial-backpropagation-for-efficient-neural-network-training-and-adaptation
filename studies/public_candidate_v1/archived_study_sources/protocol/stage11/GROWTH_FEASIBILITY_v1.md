# Stage11A: small-to-large feasibility - 2026-10-22

The author approved exploring idea 1 first: start small at full spatial resolution,
then increase model capacity. This is the first, inexpensive mechanics/cost step.
It is not a training-efficacy pilot. Ideas 2--4 remain proposals for later review.
Stage9E and Stage10A remain completed failures; no frozen historical files change.

## Question and inputs

Can the existing native six-block ViT grow from hidden width96/MLP384 to
width192/MLP768 with the SAME three attention heads, input32 and patch4,
preserving its dropout-free function? Does the small model retain enough measured
operation-cost advantage to justify a later fixed-schedule growth experiment?

Use the already-seen Stage10A endpoint states from seeds1401/1402: narrow and
full RGB32. No new development seeds or efficacy trajectories are introduced.
Read the original training-only45k/5k CIFAR10 split; never construct train=False.
Verify all source/checkpoint/summary hashes and bind them in the prospective freeze.
The trained full model supplies an ordinary native-width cost control, not a teacher.

## Expansion mathematics and limits

Let D duplicate each residual-embedding coordinate into two copies. The main
embedding order is [all original channels, all original channels]. LayerNorm's
mean and variance, with unchanged epsilon, are invariant to this duplication;
copy its scale and bias. Copy patch outputs, class token and positions.

For a linear map between duplicated spaces, duplicate output rows and divide
the duplicated input columns by2. Duplicate MLP hidden units similarly and split
their outgoing weights by2. The classifier has unchanged output classes: duplicate
its input columns and divide by2; copy its output bias.

Attention requires a different channel ordering: duplicate coordinates WITHIN
each of the three heads. Double head dimension, keep head count three. Scale the
duplicated Q and K (weights AND biases) by 2^(-1/4). Their dot product doubles from
coordinate duplication, while the new sqrt(head_dimension) denominator grows by
sqrt(2); the two Q/K factors exactly cancel the remaining sqrt(2). V is duplicated
without this scaling. The output projection maps head-wise copies back to duplicated
residual coordinates and divides duplicated input columns by2.

This is a linear parameter map and preserves the function in exact arithmetic
when dropout is absent. In FP32 compare |grown-small| <=5e-5 +1e-5*|small| per logit.
Require zero changed predictions across all5,000 validation examples in EACH seed.
Also record both accuracies, loss, maximum absolute and scaled errors. The original
narrow checkpoint accuracy and loss must reproduce before interpreting the test.

Training dropout is retained at0.1. Wide and narrow stochastic masks differ; their
train-mode predictions and subsequent optimizer trajectories are NOT claimed equal.
Independent dropout can break duplicated-unit symmetry; check this explicitly.

No Adam state transplant is claimed mathematically valid. All three diagnostic
paths start with fresh AdamW. Any future efficacy study must make optimizer reset
part of the tested treatment, include suitable reset controls, and compare with
ordinary uninterrupted training. The current step does not choose a reset schedule.

## Checks and disposable updates

Artificial CPU/CUDA tests cover nonzero classifiers, native six-block prediction
and input-derivative preservation, parameter-gradient pullback under the linear
map, a negative control with deliberately wrong attention scaling, unknown-key
rejection, fresh optimizer state, finite full updates, duplicated-unit divergence
and bitwise repeatability of the same growth/update sequence.

For each real source seed, check all validation examples before any disposable
update. Then perform8 sequential trainability updates per path (small/grown/full)
on disposable copies, with matched input/MixUp streams, ordinary full gradients,
AdamWlr.001/wd.05, smoothing.1 and MixUpalpha.2. Check all parameter counters,
weights/gradients/moments finite, model state changed, and grown patch-embedding
copies diverged by >1e-8. Save the disposable final checkpoints for independent
counter/shape/finiteness audit. These8 updates are not a convergence experiment;
neither loss decrease nor validation improvement is a pass criterion.

## Operation-cost measurement

Strict Stage10 execution: one CPU thread, deterministic errors, FP32, no TF32,
no AMP/compile, math-only attention, fixed per-operation dropout seed. This makes
controlled comparisons reproducible on the installed runtime; deployment claims
would later require comparison against optimized ordinary training too.

36 predetermined batch256 probes per seed, each sampled without replacement
within the batch from the training subset. Across probes examples may repeat.
For each probe all paths receive the same indices, augmentation and MixUp inputs.
Cycle/reverse all six path permutations equally to balance positions per seed.
Record actual MixUp coefficient and dropout seed; independent audit regenerates
these and the index stream. Hardware telemetry is sampled outside operation timers.

Restore each path's exact initial weights and zero Adam moments/counter before
EVERY measured operation, outside its timer. A separately recorded first operation
includes lazy optimizer allocation. Then16 disposable warmups per path allocate
and warm the kernels. Record36 synchronized complete operation timings per path:
index transfer, augmentation, normalization, MixUp, seed reset, forward, scalar loss
read, backward and AdamW step are inside the timer. CPU validation/audit copies,
restoration and result-file writing are outside these operation-only timers.
This is a fresh-state operation benchmark, not an end-to-end training clock.

Record growth construction/device transfer/parameter mapping/loading time separately.
The cold step also records optimizer allocation cost. A future pilot must charge
all growth, initialization, validation, logging and adaptation costs when measuring
time to quality; the warm operation ratios are not that result.

For BOTH seeds require mean(small operation time)/mean(full operation time) <=0.85
and mean(grown operation time)/mean(full operation time) <=1.10, along with
preservation and trainability, to label "feasible for pilot review". Thresholds
are chosen engineering screens, not significance tests. All failures are retained;
no automatic tolerance relaxation, alternate mapping or extra timing grid.

## Audit, output and next decision

Real output: results/stage11A_growth_feasibility_v1. Completed configuration and
attempt files are hashed; read-only re-audit on rerun. Interrupted diagnostics
restart into a new attempt, preserving all earlier attempts. Synthetic outputs
are separately named and cannot establish scientific feasibility.

Freeze code/spec/protocol/tests/launcher, input evidence and required settings
before the real diagnostic. All real input checkpoints remain unchanged.
No expanded endpoint is to be treated as a freely pretrained initialization for
a new from-scratch efficiency claim.

If feasible, the next discussion is ONE bounded small-to-large pilot with a fixed
transition, fresh development seeds and always-small/always-large comparators,
including reset controls. The pilot must ask whether recovering accuracy consumes
the earlier saving. No transition epoch, seed grid, adaptive controller or real
efficacy pilot is launched automatically by this diagnostic.

Growth is prior art: [Net2Net](https://arxiv.org/abs/1511.05641) and
[LiGO](https://arxiv.org/abs/2303.00980). Our mapping is an implementation for this
architecture, not a novelty claim. Correct expansion and cheaper small steps alone
do not demonstrate better training or justify publication. Publication stays on hold.
