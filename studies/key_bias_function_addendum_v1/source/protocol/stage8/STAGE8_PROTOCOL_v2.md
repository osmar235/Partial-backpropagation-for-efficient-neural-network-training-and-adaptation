# Stage 8 v2: bounded prospective development amendment

Recorded 2026-10-19, after the v1 development screen and before any v2 scientific
training, LGPB calibration, real freeze or evaluation seed. Machine settings:
`study_spec_v2.json`. The source review is evidence/context; this amendment
implements the author's authorized stronger-submission study.

## Reason and retained evidence

All four v1 full-AdamW runs completed 150 epochs. At LR 0.0003 the final
validation accuracies were 82.06% and 81.12% (mean 81.59%); at LR 0.001 they
were 82.92% and 83.46% (mean 83.19%). Both recipes failed the previously
declared mean >=85%, minimum >=83% readiness criteria. The final 81.12% in the
console was one lower-LR run. The RuntimeError was the intended quality stop.
No LGPB calibration or real evaluation was performed. See the separate review
and preservation manifest in `results/stage8A_review_v1/`.

Retain all v1 outputs, executed sources, specification and protocol unchanged.
Do not lower readiness, substitute best validation epochs or retroactively
declare v1 successful. The curves do not establish a cause of underperformance.
In particular, smoothed/augmented training loss is not directly comparable to
unsmoothed validation loss. No ViT LGPB result exists yet.

## One new recipe, no automatic search

Fix LR **0.001**, selected from the v1 screen. Change the horizon to **300
epochs** and add **MixUp alpha 0.2** to every training batch, including the
separately reset hardware warm-up. These are a bundled recipe revision;
subsequent improvement would not identify which ingredient caused it.
This is a development hypothesis, not a guarantee of passing readiness.

Keep the same native six-block torchvision ViT (4px patches, width192, three
heads, MLP768, dropout0.1, attention dropout0), no pretrained weights, prefix
patch projection/class token/positions/first two blocks, final four-block suffix,
native LayerNorm, AdamW defaults, weight decay0.05, label smoothing0.1, batch256,
evaluation batch512, padded crop/flip and CIFAR normalization. Use five epochs
linear LR warm-up followed by cosine decay to 1% of the base LR over the new
300-epoch horizon. Controller rho0.95 and full-after-partial rule are unchanged.

For each minibatch, after crop/flip/normalization draw one lambda from
Beta(0.2,0.2) and an independent within-batch permutation. Mix images and one-hot
labels with that same lambda. Compute ONE cross-entropy with the resulting
probability targets and label smoothing0.1; it equals the convex combination
of the corresponding two smoothed hard-label losses. No extra model forward.
Validation/test use original hard labels and no MixUp. The gate compares the
current mixed-target loss to its preceding EMA with the existing formula.

Random streams per seed are separated from dropout, shuffle and crop/flip:
NumPy default_rng(seed+96001) and CPU torch Generator(seed+97001) for warm-up
Beta/permutation; seed+98001 and seed+99001 for timed training. Reset all
training streams after hardware warm-up. Every arm has identical mixing draws
and permutations for a given seed/batch. Log lambda at every update and audit
it against the independent seed stream. MixUp cost is in augmentation timing.

## Development and stop rule

Reuse development seeds1001/1002 and the SAME stratified training-only split:
45,000 training images, 5,000 validation images, split seed80401, 500 held out
per class. This is adaptive reuse of a development split, not new independent
validation or an untouched dataset. Development does not load test images.

Run exactly **two full-AdamW baselines**, one per seed, 300 epochs/52,800
updates each. Judge FINAL validation accuracy, not best epoch. Require mean
>=85% AND each seed>=83%, unchanged. If either fails, stop and retain v2:
**no automatic third recipe, threshold reduction, seed addition or evaluation**.
If both pass, run exactly two LGPB calibration trajectories on the same split,
seeds and recipe. Mean per-epoch partial fractions form the periodic profile.
Review development before using the separate freeze/evaluation launcher.

## Prospective frozen evaluation

Reserved seeds1101--1106 are unchanged and have not been opened for Stage8.
After development review, create `FROZEN_EVALUATION_v2.json` binding completed
development hashes, all v2 source/protocol/spec hashes, LR and periodic profile.
The workflow refuses evaluation before freeze and refuses retroactive freezing.
Use separate v2 directories; changed code/settings/environment invalidate reuse.

Run all **24 trajectories**, 300 epochs/58,800 updates each, full 50k train/10k
test, with four arms and the same paired/reversed order table as v1. Match
initialization, shuffle, augmentation, MixUp and dropout streams per seed.

- Full AdamW: fresh gradients for all parameters.
- LGPB: mixed-loss gate, exact fresh suffix gradient, explicit zero prefix
  gradient; normal AdamW moments, step and decoupled decay continue.
- Periodic: loss-blind nonadjacent mask, exactly round(196*q_dev_epoch) partials,
  first update each epoch full; same continuation handling. Match development
  profile, not realized evaluation gates. Report any realized rate mismatch.
- Replay: same loss gate, prefix gradient None/state and decay skipped on
  partials, reapply preceding full-step prefix displacement; time actual overhead.

Single new primary contrast is LGPB/full AdamW: six-seed paired two-sided95%
t intervals (df5), accuracy lower bound >-1pp AND log-wall ratio upper bound <1.
These are new Stage8 criteria, separate from Stage6 confirmation. Report all
four arms vs full and LGPB vs periodic/replay with descriptive unadjusted CIs.
No superiority/equivalence claim from nonsignificance. No adding/replacing seeds
after outcomes. Always report final accuracy, uncertainty and all70/80/85%
training-only target crossings, including unreached counts and no interpolation.
CIFAR10 test images have prior project exposure; do not call them untouched.

## Timing, state and preservation

Use RTX5080 FP32, no AMP/compile/TF32. A separate196-update algorithm warm-up
is followed by complete model/optimizer/RNG reset. All arms do one training
loss.item() per update, eight explicit timed CUDA sync calls per update plus
two per epoch. Epoch wall includes indexing, shuffle bookkeeping, augmentation,
MixUp, forward, scalar extraction, gate, backward, optimizer and step logging.
Component timers do not replace epoch wall; residual is not GPU-compute savings.

Telemetry before warm-up and before/after each epoch is outside training wall;
startup median utilization>20% blocks launch. Preserve all telemetry and seeds;
drift is disclosed, not a reason for selective removal. Report training-only,
evaluation, warm-up, checkpoint and per-run elapsed costs separately. Elapsed
excludes initial data/model construction; training wall excludes validation,
telemetry, warm-up and checkpoint writes. No energy-saving claim from these data.

Retain model+optimizer checkpoints, hashes, configuration/environment, initial
state hash and all epoch/step traces. Audit prefix counters on every update for
continuation and full updates only for replay; audit identical initialization,
gate decisions, LR, MixUp lambda, periodic mask and timing totals. Matching
completed runs skip; unfinished trajectories restart from initialization.

Original Stage6/7 results, v1 Stage8 and manuscript v6 are unchanged. Report
the failed development screen and this prospective amendment in eventual
supplement/release. A passing readiness guard is operational adequacy, not SOTA
or a publication guarantee. Publication packaging and verified public
identifiers remain separate tasks; no future favorable result is assumed.

## Rationale

[MixUp (Zhang et al., ICLR2018)](https://arxiv.org/abs/1710.09412) motivates
convex example/label regularization. This compact transformer recipe is our
bounded development choice; it does not reproduce a named DeiT benchmark or
implement the locality modifications of ViT-for-small-datasets. The horizon
change tests a longer schedule with regularization, not a claim that more
epochs alone would solve the v1 result.
