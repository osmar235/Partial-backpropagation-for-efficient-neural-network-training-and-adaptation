# Independent adaptation validation preparation

2026-10-29. This is a new workload, not an extension of the old quality screen.
The first executable phase qualifies the full-gradient baseline. No official
test data may be read, decoded, scored or used for decisions in this phase.

## Scientific question and chosen workload

Does a fixed periodic prefix schedule retain useful complete-adaptation savings
under competitive execution on a new dataset and a convolutional architecture?
Use Oxford-IIIT Pet, 37-way classification, an ImageNet-1K pretrained ResNet-18,
and 224-pixel inputs. This changes dataset, resolution and architecture relative
to the latest CIFAR ViT adaptation result. ResNet itself is not new to this project.
This is conditional on one public pretrained backbone, not independent pretraining
replication. Pets/ImageNet semantic or source-image overlap has not been audited;
do not call this an uncontaminated pretraining benchmark.

Official sources:
https://www.robots.ox.ac.uk/~vgg/data/pets/
https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

## First executable phase

Use only official trainval. A deterministic stratified 20% split supplies
development validation; all remaining trainval examples train the model. The
official downloader obtains archives containing test images, but this runner
instantiates only trainval and reads/decodes/hashes only its image list. It never
opens test.txt or constructs a test dataset. Download presence is not evaluation.

The one-time cache resizes each RGB image's shorter side to 256 and center-crops
256 pixels. Training applies a deterministic seeded random 224 crop and horizontal
flip from that cache; validation uses a center crop. ImageNet normalization.
This is a bounded cached-input adaptation workload; preprocessing is not free.
Download, cache creation and baseline source-freeze verification are outside the
per-trajectory clock and recorded separately. Each trajectory charges its cache
load. Report cache preparation separately and avoid claiming raw-image pipeline
or pretraining-plus-adaptation speedups. The later comparison must use this same
clock and cache convention for every arm.

Three development seeds, FP32 with TF32 permitted versus BF16 autocast. FP32
parameters/moments, FP32 cross-entropy, channels-last layout, fused CUDA AdamW,
no compilation. BatchNorm running statistics remain fixed; affine parameters
train. Each arm starts from the same public backbone and seeded 37-class head,
fresh optimizer, identical epoch permutations/crops/flips. Model has no dropout.
Deterministic algorithms, cuDNN benchmark disabled. These are explicit competitive
execution choices; no claim of globally fastest kernels. TF32 is disabled for
common-FP32 evaluation, with own-policy evaluation also saved.

Fixed 15 epochs, batch 64, LR 1e-4 backbone / 1e-3 head, one-epoch warmup and cosine
floor 0.1, weight decay 0.01. No MixUp or label smoothing. Three discarded warmup
updates, followed by exact model restoration and a fresh optimizer. There is no
learning-rate sweep or selection using candidate performance.

Readiness: FP32 common-scored final development accuracy must average at least
85% and be at least 82% for every seed. BF16 must have a deficit <=0.30 percentage
point versus the matching FP32 arm in BOTH scoring modes for every seed and a
positive complete-time saving for every development pair. These are practical
readiness checks, not inferential claims or publication findings. If readiness
fails, preserve the records and review the baseline; do not change thresholds or
automatically launch selective training. Only final epoch counts; no best epoch.

Execution order alternates across seeds. There is one timing per learning run;
these development timings are descriptive. Complete per-trajectory time starts
before cache/model loading and includes setup, warmup/restoration, all training,
epoch validation, final own-policy validation and checkpoint/logit/step writing.
Initial package imports, dataset downloading/cache creation, preflight hashes,
manifest bookkeeping after the timed work and the independent final audit are
excluded and disclosed. Report training time separately. Common source pretraining
is excluded. All parameters are active in this baseline phase.

## Subsequent selective development and independent evaluation

After reviewing a qualified baseline, implement one fixed candidate: ResNet prefix
stem + layer1 + layer2, suffix layer3 + layer4 + classifier, prefix refreshed at
global indices 0,4,8,... . Frozen-prefix and full-gradient controls accompany it.
Inactive prefix gradients are None: optimizer state and weight decay pause; the
global LR schedule continues. All arms retain the same BatchNorm policy. This
boundary is a structural half-stage choice, not an optimized winner. No boundary,
cadence or loss-gate sweep is planned; the cadence does not use an archived loss
profile. It is not the old moment-continuation algorithm.

Use a separate, disclosed development comparison to estimate paired accuracy
variability and timing feasibility. Before any official-test evaluation, freeze
its source, baseline recipe, all arms, evaluation seeds, final budget and analysis.
The desired paired-accuracy 95% CI half-width is 0.20 percentage point; use the
development standard deviation and a t-based calculation to plan independent
evaluation seeds, with an explicit practical cap of 24. Account for uncertainty
in a small development variance estimate and report the planned half-width at
the cap if the target is unattainable. This is precision planning, not a promise
of adequate power or a selection based on a favorable candidate mean. The exact
calculation and chosen count must be frozen before evaluation is available.

The planned primary comparison is periodic minus optimized full final accuracy,
with a 0.30-point noninferiority margin and paired complete-time ratios. Freezing
is a secondary comparator. Set final multiplicity and inference rules in the
evaluation freeze, using independent adaptation seeds rather than timing repeats.
Quality results remain conditional on the chosen public source and test sample.
Do not pool development outcomes with evaluation. The evaluation should proceed
under the frozen question regardless of the candidate's development mean once
baseline feasibility, implementation validity and budget are established.

Official test evaluation is not implemented or enabled in this preparation.
The current command ends with the baseline review folder. It cannot silently
progress to candidate tuning or test-set evaluation. A later outcome that fails
the stated margin is retained and reported without selecting another cadence.

## Integrity and restart

Freeze relative source hashes, runtime versions, GPU identity, data/cache hashes,
split and official pretrained-weight hash before real baseline learning. No
imports of historical project runners; this capsule is portable with its listed
dependencies and separately downloaded data/weights. Dataset redistribution is
not included. Keep the official attribution and license link with the data record.

Save checkpoints, optimizer counters, step logs, epoch scores and final logits.
An auditor verifies hashes, reconstructs sample/augmentation/LR streams, checks
every optimizer age and initial-state pairing, and rescales saved logits with an
independent NumPy calculation. It can optionally rescore final checkpoints.
Completed runs are verified and skipped; incomplete attempts remain. Restarted
studies are marked fragmented and cannot qualify timing without review. A new
attempt is not automatically chosen for favorable results. A process lock prevents
concurrent writers. Tiny CPU/CUDA smoke data are artificial and cannot qualify
the real baseline. A native ResNet CUDA smoke checks the actual execution path.
