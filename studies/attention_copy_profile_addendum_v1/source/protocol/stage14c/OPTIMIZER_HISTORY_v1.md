# Stage14C: optimizer history and token recovery — 2026-10-24

The author approved preparation and execution of the next bounded diagnostic.
The machine-readable spec and frozen source manifest are authoritative. Earlier
Stage14A/B screens, code, results and manuscript memories remain unchanged.

## Fixed design

Two existing Stage11B full ViT checkpoints (seeds 1501/1502, 17,600 updates).
Six arms per seed: full/spatial/similarity computation, each with restore/fresh
AdamW state. Twelve trajectories, five epochs, 880 retained updates each,
10,560 in total. Compression remains 65 to 33 tokens after block 3 of 6.
Every parameter remains trainable. Fresh keeps weights and optimizer group
hyperparameters, discards all moment tensors and step counters. Lazy allocation
on the first retained update is included in cost and memory. Restore loads the
entire source state. Source parameter order, shapes, age and groups are checked.

Use the Stage14B crop/flip, MixUp (.2), permutations, dropout and minibatch
streams, batch 256, smoothing .1, AdamW weight decay .05, base LR 1e-4, one
epoch linear warmup then cosine to 1e-5. The LR schedule is identical for all
arms irrespective of optimizer age. Three discarded hardware-warmup updates
precede exact reassignment of weights and assigned optimizer state; their cost
is charged. Fresh warmup must not leave allocated optimizer state for training.
All initial logits must match the paired arm exactly and Stage14A within 1e-5.

Orders are fixed before execution, with the second seed reversing the first:
full_restore, spatial_fresh, similarity_restore, full_fresh, spatial_restore,
similarity_fresh; then the reverse. No best-epoch selection or extension.
Restored arms must reproduce every Stage14B epoch's logits within 1e-5 and
each training loss within 1e-6. A failed reproduction or audit blocks scientific
interpretation and requires investigation, not an automatic relaxed tolerance.

## Primary clock and secondary measurements

Primary complete adaptation time is source-loading start through the saved final
checkpoint, predictions, curves, step/diagnostic/telemetry logs and their hashed
manifest, plus 1/12 of that launch's shared data loading/transfer time. This is
an instrumented batch-study marginal cost, not a cold standalone deployment
latency. It includes source reads, model construction, assigned optimizer setup,
initial/epoch evaluation, discarded warmup and restoration, augmentation,
forward/backward/update, finite-gradient checks, RNG logging, two diagnostic
snapshots, checkpoint writing and artifact hashing. Synchronize GPU at boundaries.

The common environment/freeze verification, idle preflight and offline audit are
outside this clock. The post-endpoint inference benchmark, final summary and
completion-marker bookkeeping are also outside it. No inference work occurs
before the adaptation clock closes. Save scoped transform/update, setup,
warmup, validation, checkpoint and total observed-arm times separately. Retain
source training and source complete-run costs; marginal adaptation savings must
not be called from-scratch speedups. Diagnostic overhead is conservatively
included, so measured benefit applies to this instrumented workflow.

Record baseline/peak/incremental CUDA allocation, reserved memory, and maximum
absolute allocated memory across measured phases. Persistent optimizer storage
is visible in absolute peaks; fresh first-step allocation is visible in its
incremental peak. CPU artificial memory is zero. Memory and post-endpoint
inference timing are secondary and cannot rescue the primary time screen.
First/last retained updates record aggregate L2 parameter change, gradient,
first moment and second moment before/after. These are descriptive measurements,
not causal isolation of second-moment staleness or a useful convergence test.

## Frozen endpoint decisions

Use final epoch 5 only. For each seed full_restore must be at least 80% accurate
and no more than 0.30 percentage points below its original full source.
The quality anchor is the maximum of original full source and all four final
full/spatial endpoints. At least one ordinary endpoint must be within 0.30
points of that anchor. Candidate similarity_fresh must also be within 0.30.

Candidate must save at least 10% complete adaptation time versus BOTH full
states and the fastest quality-qualified ordinary endpoint (full or spatial,
restore or fresh). Equivalently its time is <=0.90 times the minimum of those
comparators. Both seeds must pass reference, quality and time. A similarly
accurate, equally fast spatial control prevents a claimed similarity advantage.
Historical times are never substituted for contemporary controls. Similarity
restore is reported as an incumbent, but is not an ordinary control: this test
does not require reset itself to accelerate the identical compressed computation.

Mechanism interaction in percentage points is
(similarity_fresh - similarity_restore) - (full_fresh - full_restore).
Descriptive mechanism support requires both positive interaction AND positive
absolute similarity reset gain in each seed (numerical epsilon 1e-10 points).
Report the spatial interaction too. No p-value, generalization or second-moment
causal claim follows from these two reused states. Do not conflate the mechanism
and practical screens: either may pass without the other. Only a practical pass
justifies planning fresh confirmation; no automatic follow-up is launched.
Mixed/negative interaction closes this fixed reset hypothesis without adding
layers, reset variants, epochs or learning rates. An audit PASS means valid
records, not a scientific success. Synthetic data can never pass either screen.

## Resume and provenance

Completed arms are hash-checked and skipped. An incomplete arm restarts from
its source in a new attempt; earlier partial files remain. Per-arm launch/data
allocation provenance is stored. Abandoned attempt compute is not charged to
successful-arm algorithmic cost; report such operational waste separately.
If retained arms span launches, the automatic practical screen is withheld
because timing was interrupted; accuracy/interaction remain reviewable. Do
not rerun completed arms automatically to repair timing. Review first.
Completed-study reruns verify hashes and re-audit without changing artifacts.

The source manifest includes Stage14B code and raw reproduction evidence,
Stage14A source logits, source checkpoints, new code/spec/protocol and software
versions. Data remain the fixed development split; no CIFAR test scoring.
Preparation uses only artificial CPU/CUDA updates and read-only real source
mapping checks. No source checkpoint training or forward scoring during prep.

This is same-task checkpoint recovery with optimizer history. Token merging and
fresh optimizers are established tools, not a new algorithm claim. Wider transfer,
weights-only checkpoint applicability, LP-FT comparison, novelty review and fresh
independent evaluation remain future requirements. Publication remains on hold.
