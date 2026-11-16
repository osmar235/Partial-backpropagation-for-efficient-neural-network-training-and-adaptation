# Stage13B: qualify the full-backpropagation reference first

Prepared 2026-10-23 after Stage13A's failed reference and before new measurements.
Author explicitly prioritized reference repair over the prepared Stage14 study.
Stage13A and Stage14 source/spec/result files remain unchanged.

## Question and scope

Stage13A's one-step score-loss comparison was inconclusive: even every full
reference worsened loss. Its operation-level resource shortfall remains a valid
observation for that configuration. We have not excluded local learning.

Stage12A supplies an already successful continuation from the same epoch2
checkpoints: epoch3 accuracy 88.58% / 89.74%, versus 81.34% / 81.08% initially.
The original seed1601 minibatch losses begin 0.6210, 1.5721, 0.9148. These losses
come from different augmented batches, so the jump alone is NOT proof of
overshoot on a fixed objective. It motivates scoring a common evaluation set
over a trajectory instead of treating one-step improvement as necessary.

This is a debugging/reproduction experiment with known reference outcomes,
not a preregistered discovery or fresh confirmation. No local method is tested
until the reference is qualified and reviewed. No LR search, chosen-best
checkpoint, early stopping, test-set access or automatic continuation.

## Two arms, one fixed epoch, two saved states

Restore Stage12A lp_full warmup checkpoints, seeds1601/1602, epoch2 (704 updates).
Both arms use exactly the original epoch3 batch permutation, augmentation RNG,
architecture, fixed BatchNorm statistics, loss, AdamW groups/parameters and
scheduled learning rates. Backbone/head base rates1e-4/.001, wd.01, batch128,
96px images. Use the source schedule index704..1055, not a restarted schedule.

- **continuation:** restore the complete saved optimizer state.
- **reset:** restore the same groups, then clear only optimizer state; identical
  weights, data streams and schedule. At this source only the classifier has
  Adam moments; the backbone is fresh in BOTH arms. Thus this isolates loss of
  classifier optimizer history under the continued recipe.

Run exactly352 updates per arm, including the final72-example batch. Four
trajectories =1,408 real updates. Reverse arm order for the second source.
Warm up on disposable copies, then restore original weights, buffers, optimizer
and both RNG streams. Warmup cannot modify experimental starting state.

## Measurements and positive-reference checks

Score the same existing5,000-example development set at0,1,4,16,64,176,352
updates. Save logits/labels for independent loss and accuracy recomputation.
This is previously used development data, not a new holdout; do not use the
curves to tune the schedule or select the best endpoint.

Record every update's loss, batch size/index hash and learning rates. Save exact
batch index streams, augmentation RNG endpoints, model hashes, optimizer ages,
final model/optimizer checkpoints, fixed-BN checks and telemetry. Time is
descriptive only: extra evaluation and recording make this unsuitable for a
resource comparison with Stage13A or original Stage12 timings.

For continuation, compare ALL352 training losses with original Stage12 epoch3
using atol=rtol=1e-5. Reproduce initial and epoch3 validation accuracy within
1e-12 (identical correct-count) and CE within1e-5. Scheduled rates and batch sizes
must reproduce the source trace. The reference qualifies only if reproduction
passes AND endpoint development CE decreases and accuracy increases relative
to the source in BOTH seeds. These are known-outcome verification criteria.

If reproduction fails, retain complete output and flag the mismatch for review;
do not silently relax tolerances, rerun until favorable, change checkpoint or
start local training. Protocol/hash/state corruption stops execution.

The reset arm has no pass/fail quality threshold. Report its complete curve,
endpoint and difference from continuation. A difference isolates the reset's
effect here; it does not prove that reset explains all of Stage13A, which also
changed the schedule, batches and scoring objective. Improvement over an epoch
does not guarantee improvement after one step. Never require monotonic losses.

## Following review

If the reference qualifies, separately freeze a matched multi-step local/full/
connected-auxiliary/suffix comparison, specifying optimizer-history mapping
and auxiliary calibration. Stage13B itself cannot conclude local learning is
effective or ineffective. Stage14 remains prepared and queued behind this check.

All source/checkpoint/evidence hashes are frozen before real execution. Completed
reruns verify output read-only; interrupted attempts remain and a new attempt
restarts from original states. Artificial CPU/CUDA workflows use tiny networks
and artificial data, never interpreted as scientific outcomes.
