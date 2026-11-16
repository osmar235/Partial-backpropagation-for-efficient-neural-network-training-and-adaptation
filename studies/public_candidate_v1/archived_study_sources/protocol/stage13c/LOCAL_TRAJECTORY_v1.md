# Stage13C: local objectives against a qualified full trajectory

Prepared 2026-10-23 after Stage13B review and before new outcomes. Author approved
continuing the matched multi-step experiment. Stage13A/B and Stage14 stay frozen.

## Question and fixed scope

Does the same three-block local-supervision design retain useful final-task
learning over a fixed epoch, and offer a practical cost advantage? Stage13B
showed that negative first-step improvement can precede successful adaptation.
It qualified a reference, not local learning. Here there is no first-step
positivity requirement, early stopping, best-time selection or LR tuning.

Reuse Stage12 lp_full epoch2 source states, seeds1601/1602:704classifier updates,
fresh backbone optimizer state. ResNet18 at96px, original45,000/5,000 development
split, fixed BatchNorm statistics. These are reused development states/data, so
this is exploratory engineering evidence, not independent confirmation.

## Arms and matched training

- full: original final-label CE through all blocks; auxiliary heads unused.
- full_aux: sum of both local-head CEs and final CE; gradients remain connected.
- local: same three CEs, detach at the two block boundaries; each local loss
  trains its own block/head. All optimizer steps occur after all backwards.
- suffix: final CE updates layer4/classifier; earlier blocks have no gradient,
  optimizer-state creation or weight decay.

Block0=stem/layer1/layer2; block1=layer3; block2=layer4/classifier. Auxiliary
GAP-linear128->10 and256->10, loss weights1. No parallel execution claims.
The implementation is the Stage13A local method, not a reproduction of DGL
or the complete local-error-signal recipes cited in Stage13A.

All arms start from identical backbone/final-classifier weights and restore
the source classifier Adam moments and step704. Backbone moments are absent.
Use original Stage12 epoch3 batch permutation and augmentation RNG, scheduled
rates at indices704..1055, wd.01, base backbone/head LR1e-4/.001, batch128 with
last batch72. Exactly352updates per arm:8trajectories,2,816updates. Reverse arm
order for seed1602. Warmup uses disposable states, followed by full restoration.

## Auxiliary calibration and optimizer history

Reproduce Stage13A's128head-only calibration batches per seed, with its saved
training-pool index stream, initialization seeds and augmentation seeds. These
2,000training examples are separate from validation, but may appear in the
subsequent original full-epoch training stream. They are not an independent
generalization holdout. Backbone and final classifier remain unchanged.

Save calibration losses, weights and optimizer state. Compare the resulting
model tensor hash and auxiliary optimizer state against the existing Stage13A
calibrated checkpoint; a mismatch stops for investigation, not retuning.
Both full_aux/local restore these calibrated auxiliary moments (age128).
Their auxiliary LR follows the same scheduled head LR during the trajectory.
Full/suffix have the shared auxiliary weights in the container but no auxiliary
optimizer state or computation. Stage12 parameter groups are retained exactly;
auxiliaries form a separate group. Mapping is checked by parameter identity/name
and tensor equality. This differs from Stage13A's deliberate zero-moment probes.

Calibration executes once per source for identical starting states. Charge its
ENTIRE measured setup cost separately to EACH auxiliary-loss arm in practical
comparisons; do not split it between them or treat reuse as free. Report it
separately too. Calibration time includes local-model/head construction,
head-only updates/transforms, state copies and calibrated-checkpoint saving;
source checkpoint loading is common and excluded. Total256 calibration steps
are additional to the2,816trajectory updates. No calibration tuning.

## Measurements and reference verification

Evaluate final-classifier logits on the same5,000development examples at
0,1,4,16,64,176,352updates. Save logits and labels. Initial and full endpoint
accuracy must reproduce original Stage12 exactly (tolerance1e-12) and CE within
1e-5; all352full training losses must match within atol=rtol=1e-5 in both seeds.
Reference reproduction AND endpoint accuracy/loss improvement are prerequisites
to interpreting candidate flags. Mismatch preserves output but blocks promotion.

Record batch indices/hashes, rates, per-loss traces, optimizer ages, source and
final states, augmentation RNG endpoints and finite gradients. Fixed BN and
inactive parameters must remain unchanged. Save final weights and Adam state.
Independent audit reconstructs streams/ages, verifies frozen inputs and summary
arithmetic. No CIFAR10 test set or adaptive choice of scoring checkpoints.

## Timing and memory definitions

For every update synchronize CUDA and record two clocks: device-ready operation
(all forwards/backwards, gradient zeroing, scalar readback, Adam step) and a
training segment also including indexing, augmentation and rate assignment.
Primary charged training time=sum(training segments)+whole calibration overhead
for full_aux/local; full/suffix have zero calibration charge. Evaluations,
warmups, loading/restoration, post-update gradient audits, logging and final
checkpoint IO are excluded and this is not end-to-end user elapsed time.
Record arm elapsed time separately so those exclusions remain visible.

Before each update, clear gradients, transform data, synchronize, record baseline
allocated bytes and reset peak counters. Report operation absolute allocated
peak, incremental peak above baseline, and reserved peak. Include first-update
Adam-state allocations: do not discard the first step or preallocate moments
for the measured trajectory. Primary memory=max incremental peak over all352
steps; report max absolute allocation too. These are GPU-operation measurements,
not total system memory. Calibration peaks/times are reported separately; a memory
pass does not imply calibration fits a smaller device. Strict deterministic FP32
with existing math-attention settings; no parallel process or energy claims.

## Prospective review filters

Controls=full/full_aux/suffix. For each seed, local must improve both endpoint
CE and accuracy over its own starting state and finish within0.30percentage
points of the BEST control accuracy. Qualifying controls are those within0.30pp
of that best accuracy. Local must additionally offer EITHER:

1. >=10% lower charged training time than BOTH full and the fastest qualifying
   control; OR
2. >=20% lower maximum incremental operation memory than BOTH full and the
   lowest-memory qualifying control.

Both seeds must pass the same defined per-seed filters to qualify for review.
Report whether time or memory supplies the advantage; a memory-only pass is
not a speedup claim. Timing samples from one trajectory per arm/state are
descriptive, not hardware-population inference. Calibration memory may negate
whole-workflow memory savings, which must be assessed explicitly in review.

These thresholds are fixed engineering filters for a short development screen,
not final convergence or acceptable long-run accuracy guarantees. Report every
arm and intermediate curve even if filters fail. No automatic extension or
best-arm substitution. Stage13A resource results retain their original scope.

## After review

A pass motivates independent confirmation or a separately frozen longer pilot;
a failure parks this tested local recipe, not all local-learning methods.
Stage14 stays queued. Preserve all original files; completed reruns audit
read-only, interrupted attempts restart from original states in a new attempt.
Artificial CPU/CUDA tests precede the source/spec/checkpoint freeze. Publication
remains on hold; this experiment makes no generic method novelty claim.
