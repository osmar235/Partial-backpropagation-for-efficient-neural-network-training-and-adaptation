# Pet selective development review — 2026-10-30

Integrity PASS. All 18 trajectories (six seeds, three arms) completed in one
unfragmented launch. The source/data/runtime freeze matches; all 36 checkpoint
rescores reproduced saved logits exactly. File hashes, optimizer ages and finite
moments, frozen-prefix identity, unchanged BatchNorm buffers, paired initial
models, data/crop/flip streams, LR schedule and step logs passed the audit.
The review ran no training and preserved every scientific result file. Only
the trainval development split was scored; official-test evaluation is unopened.

## Results

| Arm | Mean common-FP32 accuracy | Mean complete time | Mean paired complete saving | Mean accuracy difference vs full |
|---|---:|---:|---:|---:|
| Full BF16 | 92.1196% | 21.022 s | Reference | Reference |
| Frozen prefix | 92.3007% | 13.499 s | 35.77% | +0.181 pp |
| Periodic r4 | 91.7799% | 15.359 s | 26.92% | -0.340 pp |

| Seed | Frozen accuracy difference | Periodic accuracy difference | Frozen complete saving | Periodic complete saving |
|---|---:|---:|---:|---:|
| 2701 | +0.272 pp | +0.000 pp | 37.89% | 29.70% |
| 2702 | +0.272 pp | -0.543 pp | 35.10% | 26.29% |
| 2703 | -0.543 pp | +0.272 pp | 35.64% | 26.50% |
| 2704 | +0.272 pp | -0.679 pp | 35.62% | 26.55% |
| 2705 | +0.679 pp | +0.272 pp | 35.12% | 26.23% |
| 2706 | +0.136 pp | -1.359 pp | 35.24% | 26.24% |

Complete-time savings are arithmetic means of the six paired savings, not a
ratio of pooled times. They include cache/model loading, warmup/restoration,
training, validation and artifact writing under the prespecified clock. All
arms use optimized BF16 execution. Mean training-only savings are 43.20% for
freezing and 32.40% for periodic r4. Complete-time ranges are 35.10–37.89% and
26.23–29.70%, respectively; every seed saves time. These are single-session,
single-GPU development measurements with one timing per trajectory, not timing
replication across sessions or hardware. Download/pretraining/cache preparation
are excluded. Do not multiply these percentages by earlier timing results.

Freezing gives the more favorable observed tradeoff: +0.181 pp mean accuracy
difference and a shorter complete time than periodic updating in every seed.
Periodic r4 has a -0.340 pp mean difference, just below the planned -0.30 pp
margin; individual differences span -1.359 to +0.272 pp. Its accuracy cost is
not sufficiently controlled by this development evidence. A mean below the
margin is not a definitive statistical finding of inferiority.

Exploratory two-sided paired-t 95% intervals for accuracy differences are
[-0.239, +0.601] pp for freezing and [-1.013, +0.334] pp for periodic r4. These
six development seeds share one validation sample and one public pretrained
backbone. They do not establish confirmatory noninferiority, a gain in accuracy,
or broad superiority of either method. Own-policy mean differences are +0.181
pp for freezing and -0.362 pp for periodic; the qualitative interpretation is
unchanged. Simple freezing is an existing strategy, so its success here is an
empirical comparison and scope finding, not a new algorithmic invention.

## Evaluation precision planning

Periodic-minus-full development SD is 0.641615 pp. For a two-sided paired-t
95% interval with desired half-width 0.20 pp, the plug-in calculation requires
42 independent adaptation pairs. To account explicitly for uncertainty in six
development pairs, use a one-sided 95% chi-square upper bound on the SD:
s_upper = s * sqrt(5 / chi2.ppf(0.05, 5)) = 1.340499 pp. That planning SD would
require 175 pairs. Both exceed the previously stated practical cap of 24.

At 24 pairs, projected half-width is 0.270930 pp using the point SD and
0.566043 pp using the upper SD. These are planning projections, not guarantees;
normal paired differences are assumed, and validation variability may not
transfer to official-test accuracy. Code and exact values are in analysis.json
and scripts/analyze_pets_selective_results.py (NumPy/SciPy).

Recommendation: retain the cap and plan 24 independent evaluation seeds, openly
reporting that the 0.20 pp precision goal is unattainable under these projections.
Do not weaken the 0.30 pp margin or present the capped design as guaranteed to
resolve noninferiority. Final evaluation source, seeds, multiplicity and inference
remain to be frozen before test access. This review is not that execution freeze.

## Decision

Proceed toward the independent evaluation with all three prespecified arms and
the same boundary, cadence and BF16 recipe. Keep periodic-versus-full as the
planned primary comparison and freezing as the secondary control; do not switch
the primary question to the favorable development arm. Preserve unfavorable
periodic outcomes. There is no cadence/boundary sweep or development/test pooling.

The result strengthens the question of when prefix adaptation is necessary:
both selective policies save additional time over optimized full gradients, but
periodic updating has not shown an advantage over simple freezing here. The
independent evaluation should quantify that distinction, regardless of whether
it supports the original periodic method. Manuscript v10 remains unchanged until
the evaluation and its interpretation are available.
