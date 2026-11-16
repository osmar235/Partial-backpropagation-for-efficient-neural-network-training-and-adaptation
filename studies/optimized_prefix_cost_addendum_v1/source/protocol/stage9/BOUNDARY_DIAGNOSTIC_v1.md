# Stage9C: shared-state boundary/utility diagnostic

Prepared2026-10-20 before new seed1201/1202 training. The author explicitly
requested testing different prefix/suffix cut points and runnable CMD code.
Machine settings: `boundary_diagnostic_spec_v1.json`. This extends the research
reset, not the completed Stage8 freeze or paper's evaluation outcomes.

## Bounded design

Train TWO full-AdamW development trajectories, seeds1201/1202, for30 epochs
on the existing stratified45k/5k training-derived split. Retain the native
six-block ViT and common Stage8 v2 recipe: LR0.001, MixUp0.2, weight decay0.05,
label smoothing0.1, batch256, five-epoch warm-up. Use the FIRST30 epochs of the
300-epoch cosine schedule, not a compressed30-epoch decay. No85% readiness
gate applies to these deliberately early trajectories. No test dataset is loaded.

Save shared model/optimizer states at epochs1,5,15,30. All cuts and branches
start from the SAME state and use matched probe images, targets, dropout and
validation minibatches. Probes occur after checkpoint collection; they do not
alter the training trajectory or consume its random streams. Six independent
boundary-trained models are NOT required for this diagnostic.

Cuts0--5 mean tokenization/class token/position parameters plus the first
0--5 attention blocks in the prefix. The suffix always retains at least one
block, final LayerNorm and head. The old cut2 is included. This evaluates
locations in the failed ViT recipe; it does not establish a best cut for ResNet.

## Probe measurements

At every state use64 probe batches (128 training images,64 validation images
per probe). Half calibrate, half score. Training and validation role pools are
disjoint and sampled without replacement within a checkpoint. Different
checkpoints reuse the development pools; report results by seed/state/cut.

Every sample includes a normal monolithic full backward/update reference.
For each cut, compute suffix backward and its boundary activation gradient,
then extract four deployable features: raw loss, loss minus exact smoothed
target entropy, boundary-gradient RMS, and RMS(activation*boundary gradient).
Feature extraction, reduction/scalar synchronization are timed. Retain the
prefix graph and continue backward ONLY to obtain diagnostic oracle labels
and the split-full cost. This correctness path must reproduce the normal
forward and full gradients under matched train-mode dropout.

Branch from the same original model/AdamW state into full, explicit-zero
prefix and None-gradient prefix updates. Record norm of the omitted prefix
displacement relative to full, and validation CE(omission)-CE(full). Positive
validation delta means immediate harm from omission; it is not a prediction
of final trained accuracy. All branches retain exact fresh suffix gradients.
Zero continues prefix moments/step/decay; freeze skips prefix state and decay.
Use the schedule's NEXT-update learning rate for each counterfactual branch.
Non-capturable AdamW step counters remain on CPU when loading checkpoints;
moving them to CUDA could introduce per-parameter synchronization artifacts.

Time forward, suffix backward, all four features, prefix backward, gradient
assignment and optimizer work, plus common input augmentation. Model/state
restoration, diagnostic gradient cloning, validation/oracle scoring, telemetry
and excluded warm-up are outside deployment-cost estimates. The actual
diagnostic elapsed time is recorded separately. Counterfactual/oracle work
is not portrayed as a free deployment feature.
The normal full reference precedes each probe's rotating/reversing cut order;
cut-order position and GPU telemetry are retained. This is not a balanced
end-to-end policy benchmark, and a later pilot must verify projected gains.

For each feature, fit low-score thresholds at calibration quantiles0.2/0.4;
apply unchanged thresholds to scoring batches. Compare selected update error
and validation effect with equally spaced blind subsets of the SAME count
and the expected uniform-random subset mean. Also report an oracle subset,
clearly unavailable for deployment. No scoring-window sorting defines the
candidate threshold. Report realized eligibility fractions; these are one-step
eligibility decisions at reset states, not a complete online/no-adjacency policy.
Do not reverse feature directions or automatically promote a winning cut.

Cost projections at20%/40% omission mix measured split-full and partial costs
and compare normal full steps. They are microstep estimates, NOT measured
end-to-end training speedups. Feature costs include the whole four-feature
bundle conservatively; any cheaper individual implementation needs a new
timing check. Startup utilization is guarded; all telemetry and seeds retained.

## Review and stopping

Report all cuts, states, modes and features, including failures. Two seeds,
repeated states and small one-step validation probes do not establish
confirmatory significance, long-run accuracy, novelty or a safe boundary.
A candidate must demonstrate both useful predictive information and positive
GPU-cost headroom before a separately specified controlled training pilot.
No new24-run sweep or public submission is authorized by a favorable proxy.

Sources/settings/environment are fingerprinted. Completed seed training and
complete checkpoint probe blocks are verified/skipped on rerun. An interrupted
training/probe block restarts as a NEW attempt; incomplete attempts remain.
Matching completed work is never selectively rerun for a favorable result.
Changed settings/code fail and require a new version/output directory.

Output: `results/stage9C_boundary_diagnostic_v1/`, including training traces,
shared checkpoints, all probe rows, calibration/scoring comparisons, projected
costs, telemetry, configuration/hashes and `REPORT.md`. Artificial CPU checks
remain under a separate `stage9_sanity_boundary_v1` namespace.
