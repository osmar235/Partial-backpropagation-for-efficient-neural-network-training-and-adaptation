# Oxford Pet baseline review — 2026-10-29

PASS: the full-gradient baseline qualifies under every prespecified readiness
criterion. Six 15-epoch trajectories (three development seeds, two precision
policies) completed in one unfragmented launch, each with 690 optimizer steps.
Both official archives match their expected MD5 checksums. No training was rerun
in this review and no official-test images were scored.

| Seed | FP32 common accuracy | BF16 common accuracy | FP32 complete s | BF16 complete s | BF16 saving |
|---|---:|---:|---:|---:|---:|
| 2601 | 92.3913% | 92.3913% | 34.545 | 22.305 | 35.433% |
| 2602 | 91.9837% | 92.9348% | 33.156 | 21.661 | 34.668% |
| 2603 | 92.7989% | 92.9348% | 33.276 | 21.305 | 35.974% |

Common-FP32 validation means: 92.3913% FP32/TF32 and 92.7536% BF16.
Mean complete adaptation times: 33.659 s and 21.757 s, respectively.
Mean training-only times: 28.769 s and 17.414 s. BF16 complete-time savings
range from 34.668% to 35.974% across the three matched development pairs.
These are descriptive development timings, with one timing per trajectory.

The FP32 baseline exceeds the fixed 85% mean and 82% per-seed floors. BF16's
worst common-score deficit is zero; its worst own-policy deficit is 0.27174
percentage point (two of 736 validation examples), within the fixed 0.30-point
screen. This screen is not statistical proof of equivalence. All three BF16
pairs have positive complete-time savings. No thresholds or recipes changed.

The review reverified the frozen source, data/cache/weight hashes, trainval
images and annotations, deterministic split, completed artifact hashes, logits,
optimizer ages, data/augmentation/LR streams, paired initialization, and unchanged
BatchNorm buffers. The saved original audit contains 12 checkpoint rescoring
checks with zero maximum logit error; this review did not repeat those GPU
forwards. All 45 baseline files remain byte-identical (excluding the process lock).

Timing scope includes each trajectory's cache/model load, warmup/restoration,
training, validation and checkpoint/logit/step writing. One-time cache and weight
preparation was 8.921 s and excluded. The metadata's approximately 0.009 s download
entry only describes the later already-local dataset check: it excludes the
separate recovery downloader, failed transfers and wall-clock waiting. Do not
report it as the actual download duration or claim a raw-image end-to-end speedup.

## Decision and next experiment

Use qualified BF16 execution for every arm of the next development comparison:
full backpropagation, permanently frozen prefix, and fixed periodic prefix
refresh at global updates 0,4,8,... . The prefix is stem + layer1 + layer2;
the suffix is layer3 + layer4 + classifier. Inactive prefix gradients are None,
so its AdamW state and decay pause while the global LR schedule continues.
Keep fixed BatchNorm statistics, identical initializations, paired data streams,
and the same complete-time clock. Do not sweep cadence or boundary.

This next runner still needs implementation and prospective freezing; no new
long-run command is issued by this review. Use only trainval development data.
Use the development paired variability for the planned evaluation precision
calculation (target 0.20 pp half-width, practical cap 24), accounting for variance
uncertainty. Freeze independent evaluation seeds, budget and inference before
official-test evaluation. Preserve the planned 0.30 pp noninferiority margin.

These savings come from BF16 full-gradient execution, not from the proposed
selective method. They establish the stronger comparator that selective training
must improve upon. They do not yet provide an independent positive replication
of periodic updating. Conclusions remain conditional on one public pretrained
backbone, this trainval split and this GPU. Manuscript v10 remains unchanged.
