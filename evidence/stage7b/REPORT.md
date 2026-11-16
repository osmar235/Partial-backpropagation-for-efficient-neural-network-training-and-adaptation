# Stage 7B mechanism ablations — independent review

Reviewed 2026-10-14. Six new CIFAR-10 seeds 51–56; six variants; all 36 runs
complete. This is post-confirmation evidence and does not alter the original
Stage 6B analysis or its inferential gates.

## Finding

The training-efficiency result is supported again under Stage 7 instrumentation,
but the loss gate's superiority is not demonstrated. Matched random and periodic
skipping achieved very similar accuracy and time. Adam moment continuation
improved accuracy relative to freezing the prefix, but no clear advantage over
displacement replay was established. The paper should report all of these
outcomes and narrow its mechanism claims accordingly.

## Audit

The submitted ZIP passed its CRC check and all 119 file contents matched the
local result folder. Configurations, source hashes, fingerprints, six rotated
execution orders and completion markers were verified. All 211,680 logged
steps and 1,080 epoch records were checked: EMA history, full/partial decisions,
exact random/periodic masks, cosine learning rates, counts, backward-mode
timers, component totals, final metrics and time-to-target crossings.

Every run has 30 epochs and 5,880 updates. The planned training settings and
196 excluded warm-up updates are recorded in the configurations. Their actual
execution is supported by source provenance; warm-up has no separate raw trace.
No final parameter checkpoints were supplied, so a trace audit cannot independently
reconstruct learned weights or inspect optimizer states.

All ten reported paired estimates and their intervals were independently
recomputed from individual run summaries. All 218 recorded frozen Stage 6 and
release files remained unchanged. Machine-readable checks are in `audit.json`.

## Descriptive results

Means across six seeds. Wall reduction is calculated from the geometric mean
paired time ratio against Adam, not from the ratio of arithmetic mean times.

| Variant | Mean final accuracy | Mean training seconds | Wall reduction vs Adam | Mean partial updates |
|---|---:|---:|---:|---:|
| Adam | 84.902% | 401.47 | reference | 0% |
| LGPB | 84.793% | 367.24 | 8.52% | 35.930% |
| Random matched | 84.798% | 368.09 | 8.31% | 35.986% |
| Periodic matched | 84.833% | 369.20 | 8.04% | 35.986% |
| Prefix frozen | 84.217% | 364.33 | 9.25% | 35.468% |
| Displacement replay | 84.972% | 369.39 | 7.99% | 35.833% |

Paired contrasts below use **LGPB minus comparator** for accuracy and
**LGPB/comparator** for time. Intervals are exploratory paired 95% t intervals
with five degrees of freedom, unadjusted for multiple comparisons.

| Comparator | Accuracy difference, pp (95% CI) | Training-wall ratio (95% CI) |
|---|---:|---:|
| Adam | -0.108 (-0.580, +0.363) | 0.9148 (0.9101, 0.9194) |
| Random matched | -0.005 (-0.319, +0.309) | 0.9977 (0.9925, 1.0029) |
| Periodic matched | -0.040 (-0.322, +0.242) | 0.9947 (0.9869, 1.0026) |
| Prefix frozen | +0.577 (+0.219, +0.935) | 1.0080 (0.9980, 1.0181) |
| Displacement replay | -0.178 (-0.449, +0.092) | 0.9942 (0.9854, 1.0031) |

LGPB was faster than Adam for all six seeds; accuracy differences ranged from
-0.68 to +0.68 pp. The interval for mean accuracy includes zero. This is not a
new formal non-inferiority declaration, and intervals containing zero do not
establish equivalence. Against frozen-prefix updates, LGPB was more accurate
in all six seeds, with about 0.8% greater training wall time on the geometric
paired estimate. Its time difference from freezing remains uncertain.

## Timing interpretation

Within LGPB, across-seed mean full-backward duration was 39.936 ms and partial
backward duration was 22.906 ms, approximately 42.64% shorter. These empirical
mode timers include gradient assignment/zero construction in the backward
region; they are not pure GPU-kernel measurements. The partial rate was 35.93%,
so this mode difference is not a 42.64% reduction in overall training time.
The measured overall reduction against Adam is 8.52% (exploratory transformed
95% interval 8.06%–8.99%).

LGPB's mean wall-minus-component overhead was 15.70 seconds; Adam's was 15.77
seconds. Report it as synchronization/loop/logging overhead, not GPU compute
savings. All six seeds of every variant reached the logged 80% target; no
favorable target or seed should be selected for the headline conclusion.

Absolute Stage 7 times differ from Stage 6 and must not be directly pooled or
used to infer regression. Instrumentation and system conditions differ. Keep
the original nine-seed 10.24% result and this new six-seed 8.52% result separate.

## Consequences for the paper

- Retain the measured workload-specific efficiency claim.
- Report that loss-informed decisions did not show a clear advantage over the
  two archived-rate-matched blind schedules. Do not claim the loss signal was
  proved necessary, or that the methods are equivalent.
- Report an accuracy benefit for moment continuation compared with prefix
  freezing, with exploratory uncertainty and multiplicity disclosed.
- Do not claim unique superiority of Adam moment continuation: displacement
  replay remained competitive and had numerically higher mean accuracy.
- These are whole-algorithm ablations: freezing/replay change subsequent losses
  and gate decisions, and blind schedules use archived epoch counts. They do
  not constitute a common-decision-trace causal decomposition.
- No controller tuning, new variants, replacement seeds or revised inferential
  gates are introduced after seeing these results. Seeds 51–56 are now seen.

## Next action

Proceed with the already planned **Stage 7D CIFAR-100 transfer**, seeds 63–68,
without changing the controller, learning rate or protocol. Then run Stage 7C
sensitivity, seeds 57–62. Stage 7E remains optional. Generalization would expand
the demonstrated workload scope but would not establish loss-gate superiority.
Any future common-mask or stronger gate-identification study needs its own
prospective plan and new seed block; it is not added retrospectively here.

Reproduce this independent audit:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\projects\.venv\Scripts\python.exe" -B scripts\analyze_stage7b_review.py
```

It reads the returned ZIP and original records and writes only the separate
`results/stage7B_review_v1/` review outputs. The submitted results are preserved.
