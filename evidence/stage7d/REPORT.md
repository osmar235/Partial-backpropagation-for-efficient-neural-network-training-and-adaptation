# Stage 7D CIFAR-100 transfer — independent review

Reviewed 2026-10-17. Six seeds 63–68, Adam and the unchanged LGPB controller:
all 12 runs complete. This is a post-confirmation extension, separate from the
original Stage 6B confirmation and its inferential gates.

## Finding

The wall-time saving transferred to CIFAR-100, with an observed accuracy cost.
LGPB reduced training wall time by 9.98% but lost 0.695 percentage points of final
accuracy on average. These results support a dataset-transfer speed–accuracy
trade-off, not a claim of preserving accuracy or a universally superior method.

## Independent audit

All 70,560 step records and 360 epoch records were checked for the planned
settings, source hashes, configuration fingerprints, alternating algorithm order,
loss EMA and full/partial decisions, cosine rates, mode counts, timing totals,
final evaluation metrics and time-to-target crossings. Every run contains
30 epochs and 5,880 updates. The parameter counts agree with the unchanged
stem+layer1 prefix and a 100-class ResNet-18-GN output head. All reported paired
effect estimates and confidence intervals were independently reproduced.

All 218 previously recorded Stage 6/release files retained their hashes.
An input hash manifest is in `audit.json`. The audit reads the existing local
results directly; no Stage 7D ZIP was needed. It writes only to this separate
review folder. No training code, output files or prospective settings changed.

Source/configuration records support the intended implementation and excluded
196-step warm-up; traces cannot independently verify actual optimizer states
or learned parameters, and warm-up has no separate raw trace/checkpoint.

## Results

| Quantity | Adam | LGPB |
|---|---:|---:|
| Mean final test accuracy | 53.857% | 53.162% |
| Mean training wall time | 405.01 s | 364.54 s |
| Mean partial-update fraction | 0% | 35.785% |
| Mean full-step backward region | 40.301 ms | 39.979 ms |
| Mean partial-step backward region | — | 22.740 ms |
| Mean wall-minus-component overhead | 14.15 s | 13.78 s |

Paired accuracy difference, LGPB minus Adam: **-0.695 pp**, exploratory paired
95% t interval **-1.358 to -0.032 pp**. LGPB was less accurate in five of six
seeds; differences ranged from -1.35 to +0.24 pp. Two seeds lost more than
one point (-1.20 and -1.35 pp).

Geometric paired training-wall ratio, LGPB/Adam: **0.9002**, exploratory paired
95% interval **0.8881–0.9124**. This is **9.98% less training wall time**
(transformed interval **8.76%–11.19%**). LGPB was faster in all six seeds,
with per-seed reductions ranging from 8.27% to 11.19%.

Intervals use six paired seeds and five degrees of freedom and are unadjusted
for multiple comparisons. No formal Stage 7 non-inferiority test was planned.
The accuracy interval extends below -1 pp, so even the original Stage 6 margin
would not be supported by this interval; do not describe this as passing a
transferred accuracy guarantee or retroactively apply the old pass/fail gate.

## All predeclared time-to-quality targets

Times are epoch-evaluation threshold crossings, not interpolated within epochs.
Training time excludes validation and warm-up, as in the prospective protocol.

| Accuracy target | Adam reached | LGPB reached | Geometric LGPB/Adam time ratio (exploratory 95% CI) |
|---|---:|---:|---:|
| 40% | 6/6 | 6/6 | 0.8950 (0.8387, 0.9550) |
| 50% | 6/6 | 6/6 | 0.9420 (0.9060, 0.9795) |
| 60% | 0/6 | 0/6 | not estimable |

LGPB reached the 50% target about 5.80% sooner on the geometric paired estimate,
but this does not remove its lower final accuracy. Report the unreached 60%
target alongside reached targets rather than selecting a favorable target.

## Scientific interpretation and next step

- This adds a second dataset within the same ResNet-18-GroupNorm family,
  using the fixed 30-epoch, common-learning-rate recipe. It does not establish
  another architecture, hardware platform, or an optimized CIFAR-100 recipe.
- Report the accuracy cost alongside the approximately 10% timing benefit.
  Do not retune the controller or recipe on seeds 63–68 and still label the
  amended result independent transfer. These seeds are now seen.
- Keep the Stage 6B 10.24% timing result, Stage 7B 8.52% result and Stage 7D
  9.98% result separate. Do not pool their timings or infer a hardware drift
  mechanism from different absolute durations.
- Stage 7B still provides no clear gate advantage over matched blind schedules
  and no clear continuation advantage over replay. This transfer study compares
  only Adam/LGPB and cannot resolve those mechanism questions.
- Next is the already planned **Stage 7C sensitivity study on CIFAR-10 seeds
  57–62**, with unchanged settings. It is not a search for a replacement winning
  controller and cannot repair the observed CIFAR-100 accuracy cost.
- Stage 7E remains optional. No extra variants, replacement seeds, retrospective
  thresholds or new inferential claims were introduced in this review.

Reproduce the audit:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\projects\.venv\Scripts\python.exe" -B scripts\analyze_stage7d_review.py
```

Independent numerical estimates are saved in `independent_results.json`.
