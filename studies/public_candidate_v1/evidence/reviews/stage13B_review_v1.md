# Stage13B reviewed — 2026-10-23

**The full-backpropagation reference qualifies in both seeds.** Restoring the
original optimizer reproduces every original epoch3 training loss exactly:
352/352 per seed,704/704 total. Initial and final development scores reproduce.
The four trajectories contain1,408 updates; integrity and reference checks pass.

## The short-horizon criterion was misleading here

| Seed / arm | Initial accuracy | After 1 update | After 352 updates | Initial CE | After 1 update CE | Final CE |
|---|---:|---:|---:|---:|---:|---:|
|1601 continuation|81.34%|59.90%|88.58%|0.555629|1.428651|0.342107|
|1601 reset|81.34%|57.16%|87.84%|0.555629|1.597998|0.376746|
|1602 continuation|81.08%|64.20%|89.74%|0.557440|1.121762|0.303485|
|1602 reset|81.08%|60.46%|90.30%|0.557440|1.279234|0.287088|

All four runs worsen both accuracy and CE after the first update, yet improve
both by the fixed one-epoch endpoint. This is measured on the SAME5,000-example
development set at every time point, so it is not merely a change in minibatch
difficulty. Both seed1601 arms first recover beyond their starting scores at
the observed step16; both seed1602 arms do so at observed step64. These are the
first sampled recovery points, not exact crossing times.

The retained-moment reference also deteriorates initially. Resetting classifier
Adam history therefore cannot be the sole explanation for a harmful first step
in this setting. The exact mechanism of the transient deterioration is not
identified; these data do not isolate feature disturbance, step size or curvature.

## What resetting changes

Both arms start at identical weights, use identical batches and augmentations,
and follow the same original learning-rate schedule. Only the classifier had
optimizer history at the source; the backbone starts fresh in both arms.

Reset makes the first-step accuracy drop larger by2.74pp /3.74pp and increases
first-step CE by0.169347 /0.157472 relative to continuation. At the endpoint,
reset is0.74pp worse for1601 but0.56pp better for1602; CE is0.034639 higher /
0.016397 lower. Thus its initial harm is consistent across these two states,
while its endpoint effect is mixed. Neither a universal reset benefit nor a
universal reset penalty is supported.

The reset arm is not expected to reproduce the continuation trace. Its false
reference_qualified field in the raw comparison table is by design: only the
continuation arm is the reference. It is not a failed efficacy criterion for reset.

## Consequences for Stage13A and the paper

Stage13A remains a valid record of its measured operation costs and an
INCONCLUSIVE learning-utility comparison. Requiring immediate positive score
improvement would reject a continuation known to work over an epoch. We should
replace that selection rule in a new experiment with a fixed multi-step endpoint,
while retaining the original protocol, thresholds, raw data and interpretation.

Stage13B is not an exact replay of Stage13A: it uses the original scheduled
rates, original training stream and development scoring instead of that
diagnostic's fixed rates, probe batches and independent training score batches.
It establishes a usable trajectory reference, not a complete causal explanation
of every Stage13A failure. It does not test local learning or overturn the earlier
local resource shortfall. Published local-learning methods are not refuted.

This is known-outcome debugging on previously examined development data and
two source states. Retain the methodological lesson in the paper discussion;
do not market it as a new algorithm, fresh confirmation or population inference.

## Independent verification

- Re-ran the frozen auditor and verified completion/raw/source/evidence hashes.
- Independently recomputed all28 curve points from saved logits using float64
  cross entropy, plus stepwise trace differences and recovery observations.
- Audited batch coverage/order, augmentation RNG endpoints, optimizer ages,
  fixed BatchNorm buffers and source/final state hashes across all1,408 updates.
- Reloaded both source models and all four final model checkpoints on CUDA;
  reproduced saved development scores without optimizer updates. See
  saved_state_rescore.csv and independent_audit.json for exact logit errors.
- Preserved prior immutable project files and every raw result; final file
  counts and update verification are recorded alongside this report.

## Decision

Reference repair is complete. The next scientific step is a separately frozen
matched multi-step comparison of full, connected auxiliary, local and suffix
training, with explicit optimizer-history mapping and auxiliary-head costs.
Use a fixed trajectory endpoint rather than demanding immediate improvement.
Resource measurements must be repeated for the actual continued training setup.

That follow-up is not prepared or executed in this review. Stage14 stays queued
while this question is resolved. No automatic tuning, extra epochs, test-set
evaluation or publication action. Publication remains on hold.
