# Stage 7C sensitivity — independent review

Reviewed 2026-10-18. Six variants on CIFAR-10 seeds 57–62; all 36 runs complete.
These are exploratory post-confirmation results. They do not alter Stage 6B,
replace the frozen controller, or constitute a hyperparameter-selection study.

## Independent audit

All 211,680 step and 1,080 epoch records passed checks of configuration
fingerprints, source hashes, cyclic execution orders, variant-specific EMA
decays and partitions, gate decisions, learning-rate schedule, partial streaks,
mode counts, timing reconciliation, final metrics and all target crossings.
Each run contains 30 epochs and 5,880 updates. All five paired estimates and
95% confidence intervals were independently reproduced. All 218 recorded
original Stage 6/release files retain their hashes.

The input hash manifest is in `audit.json`; numerical results are in
`independent_results.json`. This review preserves the supplied output folder.
Configuration/source provenance supports intended warm-up and update-state
behavior, but the traces contain no separate warm-up record or learned-state
checkpoint with which to verify optimizer states independently.

## Descriptive results

Accuracy deltas are variant minus the frozen LGPB reference. Wall ratios are
paired geometric ratios against that same reference, not against Adam.
There is no full-gradient Adam baseline within Stage 7C.

| Variant | Mean accuracy | Mean partial fraction | Accuracy delta, pp (95% CI) | Wall ratio (95% CI) |
|---|---:|---:|---:|---:|
| Frozen LGPB | 84.922% | 35.933% | reference | reference |
| EMA 0.90 | 84.653% | 35.337% | -0.268 (-0.505, -0.032) | 0.9748 (0.9196, 1.0333) |
| EMA 0.99 | 84.827% | 39.606% | -0.095 (-0.385, +0.195) | 0.9840 (0.9587, 1.0100) |
| No forced-full | 84.405% | 55.711% | -0.517 (-0.768, -0.266) | 0.9438 (0.9293, 0.9585) |
| Stem-only prefix | 85.027% | 35.779% | +0.105 (-0.00031, +0.21031) | 1.0962 (1.0643, 1.1290) |
| Stem+layer1+layer2 prefix | 84.098% | 36.259% | -0.823 (-1.135, -0.512) | 0.9502 (0.9294, 0.9714) |

All intervals are exploratory paired t intervals with five degrees of freedom,
unadjusted for multiple comparisons. Do not turn an interval containing zero
into proof of equivalence, or the near-zero stem-only lower bound into a claim
of established accuracy superiority by rounding it away.

## Interpretation

- Removing forced-full refresh increased partial use by 19.78 points and reduced
  training wall time by 5.62%, while losing 0.517 pp mean accuracy. It was faster
  and less accurate in every paired seed. Maximum consecutive partial streaks
  were **60, 93, 86, 86, 59 and 70**, compared with one in the frozen reference.
  Every run completed with finite losses and reached 80%; long streaks did not
  produce numerical failure, but did accompany lower final accuracy. This
  supports the refresh cap as an accuracy–compute trade-off under this recipe,
  not as proof of universal stability or an optimal cap.
- The larger prefix saved 4.98% wall time while losing 0.823 pp accuracy;
  all six seeds were faster and less accurate. The stem-only prefix used 9.62%
  more wall time for a small, uncertain +0.105 pp accuracy change. Prefix
  boundary affects practical savings and accuracy; parameter count alone does
  not describe the backward cost.
- EMA 0.90 had a small mean accuracy loss and an uncertain timing change.
  EMA 0.99 increased partial use but showed no clear accuracy/time advantage.
  Neither result establishes equivalence or justifies replacing the frozen
  0.95 value after inspecting these data.
- Every variant reached each logged target (60%, 70% and 80%) in all six seeds.
  Stage 7C provides no within-stage comparison to Adam; do not invent one by
  pairing it with Adam runs from different seeds or phases.

## Timing limitation

Absolute timings changed substantially across seeds: frozen LGPB took 216.70 s
in seed 57 versus 354.47–375.17 s in seeds 58–62. Mean full-backward region
timings for seed 57 were about 26–27 ms across variants, compared with roughly
38–41 ms for most later runs. Variation also occurred within a seed: in seed 58,
EMA 0.90 averaged 36.15 ms for full backward versus 40.03 ms for the reference.
Within that EMA 0.90 run, epoch 1 full-backward time was 26.26 ms and epoch 30
was 41.09 ms, showing that the variation was not solely between seeds.
`timing_by_seed.csv` preserves all per-seed operation times and execution orders.

These records establish timing heterogeneity but do not identify its cause
(no independent clock/power/load telemetry was recorded). Cyclic order and
paired ratios help, but cannot remove drift occurring between sequential runs.
Retain all planned seeds and report the paired estimates with this limitation;
do not select or discard seeds to improve timing significance. Small EMA timing
differences are especially weak grounds for a mechanism claim. Do not pool
absolute timing with Stage 7B/7D or infer a cross-phase speed ranking.

## Combined project implication and next action

The core reviewer follow-up (7A, 7B, 7D and 7C) is complete: 84 new substantive
training runs. Frozen LGPB remains a defensible fixed operating point, not a
demonstrated optimum. The wider evidence must retain three boundaries:

1. Stage 7B found no clear advantage for the loss gate over matched blind
   schedules and no clear superiority over displacement replay.
2. Stage 7D transferred timing savings to CIFAR-100 with an accuracy cost.
3. Stage 7C shows additional speed comes at an accuracy cost when refreshes
   are removed or a larger prefix is skipped, with timing heterogeneity disclosed.

Keep the original Stage 6B confirmation separate. Seeds 57–62 are now seen.
No controller changes, seed replacements, retrospective thresholds or new
inferential gates are introduced after seeing these results.

The remaining **Stage 7E AdamW robustness experiment is optional**, using its
existing fixed seeds 69–74 and recipe. It can address the zero-weight-decay
limitation, but does not resolve gate superiority or the CIFAR-100 accuracy
cost. It is not required to call the core plan complete. Proceed with that
launcher if taking the optional extension, then consolidate the manuscript
around the full results. Manuscript/submission files remain unchanged here.

Reproduce this audit without training:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\projects\.venv\Scripts\python.exe" -B scripts\analyze_stage7c_review.py
```
