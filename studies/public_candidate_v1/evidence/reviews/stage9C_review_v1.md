# Stage9C independent review â€” 2026-10-20

The boundary experiment completed correctly. Larger prefixes offer computational
headroom, but the measured boundary-gradient signal has not demonstrated useful
incremental information over a cheap MixUp-strength signal. Do not promote this
diagnostic into a successful replacement method or launch a large efficacy sweep.
Publication remains on hold.

## Integrity and scope

Independent audit PASS. Both full-AdamW development trajectories completed
30 epochs (5,280 updates each), with all eight requested checkpoints and all
eight probe blocks. The 6,144 branch measurements represent 512 shared probe
batches, of which 256 are scoring batches, at eight states from TWO model seeds.
They are not 6,144 independent training experiments or eight independent seeds.

Verified all completion hashes, configuration fingerprint and 12 executed source
hashes. CPU-loaded all eight checkpoints; model parameters and Adam moments are
finite, parameter counts match, and every optimizer step counter matches its
checkpoint. Verified calibration/scoring index separation, uniqueness within
each role, batch sizes, dropout streams, paired branch metadata, timing
decompositions and freeze displacement identity. Independently recomputed all
768 signal-summary rows and 192 cost projections from raw measurements.

The original 218-file baseline, preserved Stage8 v1/v2 sources and results,
Stage9C preparation files, and manuscript v7's 13 recorded artifacts matched
their existing manifests before updating the two mutable project pointers.
Raw Stage9C and executed scientific sources are preserved; see
`preserved_inputs_sha256.csv`, `audit.json`, and `preservation_after_review.json`.

Full baselines at epoch30 reached 77.66% / 76.82% validation accuracy. These are
the first30 epochs of a300-epoch schedule, not failed final readiness tests.
No adaptive trajectory or test-dataset evaluation was run.

## Boundary cost comparison

Mean arithmetic ratios across the eight states, at a HYPOTHETICAL40% omission
fraction. Savings are relative to the measured ordinary full-update operation
cost, not measured end-to-end training or time-to-quality gains.

| Prefix blocks | Prefix parameters | Zero-gradient projected savings | Frozen-prefix projected savings |
|---:|---:|---:|---:|
| 0 | 22,080 | -4.3% | -4.2% |
| 1 | 466,944 | -1.3% | -0.4% |
| 2 (old cut) | 911,808 | 3.4% | 4.7% |
| 3 | 1,356,672 | 5.7% | 7.6% |
| 4 | 1,801,536 | 10.2% | 12.7% |
| 5 | 2,246,400 | 13.1% | 16.1% |

Cut4 comprises tokenization plus the first four of six blocks; two blocks,
final normalization and head remain in the suffix. Cut5 is the most economical
in this measurement, but does not establish the best training tradeoff. Moving
from cut2 to cut4 materially increases the work available to skip. Cuts0/1
do not offer useful headroom at the assumed rate. The split/whole-feature
pipeline itself costs approximately4.2--5.7% extra on full updates.

These probes use batch128, whereas ordinary training uses batch256. All four
features are timed together; individual-feature deployment cost is unmeasured.
The timers synchronize component boundaries and omit diagnostic restoring,
cloning and oracle evaluations. Full reference always runs before branches;
cut order is rotated/reversed. Probe thermal/clock samples span varied states.
An actual deployment benchmark at batch256 is required before accepting any
numerical saving. A blind schedule can choose before forward and avoid gate/
split overhead; it must be implemented efficiently in a subsequent comparison.

## What the original four signals show

For cut4, using the calibration40th-percentile threshold and then held-out
scoring batches, the boundary-gradient RMS selects41.8% of scoring probes on
average. This eligibility is not a realized online skip rate.

| Cut4 statistic, mean of eight states | Explicit zero gradients | Frozen prefix (`grad=None`) |
|---|---:|---:|
| Spearman correlation with omitted prefix-update error | 0.750 | 0.597 |
| Selected absolute update error / blind expected error | 0.904 | 0.978 |
| Selected validation CE delta relative to full update | +0.023351 | -0.006373 |
| Selected validation CE delta minus blind expected delta | -0.003970 | -0.001904 |
| States with smaller validation delta than blind | 6/8 | 6/8 |
| States with smaller validation delta than count-matched periodic | 6/8 | 5/8 |

Lower selected update error is real on these batches, particularly for
zero-gradient continuation. It is a parameter-displacement proxy, not a
learning-value or accuracy result. For freezing the reduction is only2.2%.
Negative validation CE deltas mean the omitted single update gave smaller
validation loss than its paired full update; they do not imply that prolonged
freezing improves accuracy. The zero-gradient and frozen branches differ
substantially in immediate validation effects, so continuation is not a
protected design choice.

Raw-loss selection at cut4 increases absolute update error by5.5% (zero) /
0.6% (freeze) relative to blind, averaged across states. Entropy-corrected loss
changes that error by less than1%. Neither supplies a convincing new gate.
Activation-weighted boundary gradients are similar to boundary RMS; the
diagnostic does not establish a useful reason to pay for both features.

Effects vary by phase. For cut4/freeze the boundary signal's mean error
correlation is -0.082 at epoch1, then0.721/0.843/0.907 at epochs5/15/30.
The selected-minus-blind validation improvement at epoch30 is only0.000291 CE.
An aggregate correlation cannot establish safe behavior throughout training.

## Additional cheap-control check â€” explicitly post hoc

After inspecting these results, we added two exploratory controls using ONLY
the existing raw data: prefer high MixUp strength `min(lambda,1-lambda)`, or
high target entropy. Both are available without a boundary backward pass.
Their directions were chosen after inspecting the diagnostic; they were NOT
part of the frozen Stage9C four-feature plan. They must be disclosed and tested
prospectively if used in a new study.

With a calibration-only40th-percentile threshold on NEGATIVE MixUp strength,
cut4's mean selected update-error ratios are0.910 (zero) /0.983 (freeze).
Its selected-minus-blind validation CE differences are -0.004598 / -0.003078.
Thus the cheap control captures much of the apparent utility. Realized counts
differ from the boundary threshold; this comparison alone is not resource matched.

For an additional equal-count INFORMATION comparison, rank the scoring window
by cheap strength and take the same number of probes selected by the boundary
threshold. This uses future score-window ranks and is NOT a deployable online
policy. Cut4's subsets overlap88.8% on average. Boundary RMS and negative
MixUp strength have mean Spearman correlation0.827. Boundary selection reduces
update error by only0.71% (zero) /0.63% (freeze) beyond these equal-count cheap
subsets. Its validation CE is WORSE on average by0.000372 /0.000232 and is
better in only3/8 states for either state-handling mode. There is no demonstrated
incremental validation benefit from the more complicated signal in this check.

These post hoc comparisons do not prove that a MixUp policy works, establish
statistical equivalence, or invalidate all gradient-based gates. They identify
an essential cheap comparator that a replacement controller must beat. No new
normalizations or residual gates were fitted to these scoring outcomes.

## Research decision and bounded next step

The author's cut-location idea is supported as a computational question:
cuts4/5 have appreciably more potential savings than the old cut2. This is not
yet evidence for an adaptive method contribution. Keep cut4 as a practical
DEVELOPMENT candidate because it has material headroom while leaving two
trainable suffix blocks; this is a design judgment, not a statistically selected
optimal cut. Cut5 remains a diagnostic alternative, not a second automatic sweep.

Do not preserve the old raw-loss gate or zero-gradient continuation by default.
Do not claim boundary-gradient novelty or superiority from the proxy correlations.
The cheap MixUp comparator currently prevents a persuasive case for paying for
the boundary decision. A full training pilot of that gate alone is not justified.

If the next question is whether any allocation rule at this boundary has a
useful advantage, the smallest informative controlled pilot should keep ONE
boundary fixed and compare full AdamW, efficient periodic, matched random,
cheap MixUp-only scheduling, and boundary-gradient scheduling. Use the same
prefix-freezing rule for all partial-update arms so the controller contrast is
interpretable; zero continuation is a separate future ablation, not another grid.
Freeze online thresholds, resource-matching rules, targets, seed handling and
stopping criteria before running. Include exact mask/count replay if realized
resources diverge. Calibration must use past data only; diagnostic score-window
sorting cannot appear in an online controller. Test any cheap policy's value
against blind controls too.

Before spending that training budget, benchmark the deployable batch256 paths
and quantify the gate's overhead relative to the efficient blind/cheap paths.
If the extra gate cost exceeds its plausible advantage, stop the boundary-gate
direction rather than add tuning knobs. A pilot would remain development;
fresh-run confirmation and a second workload would follow only a practical win.
No new efficacy run, reserved-seed evaluation, public release or submission
was launched during this review. Current project master:v6.5.

## Prepared next CMD: cost only

`scripts/run_stage9D_deployment_cost.cmd` reuses saved checkpoints15/30 from
both development seeds at cut4, batch256. Its512 timing rows compare ordinary
full, predecided prefix-no_grad partial, boundary-RMS full and boundary-RMS
partial paths. No new trajectory is trained. CPU artificial paired gradients,
parameters, freeze counters and summary arithmetic passed. Actual CUDA checks
run before real measured samples. See the separate Stage9D protocol/validation.
Scientific Stage9D has not been executed during preparation.

## Reproduction

The review script imports no executed Stage9C analysis functions. Full audit:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\projects\.venv\Scripts\python.exe" -B scripts\review_stage9C.py --check-checkpoints
```

No training is performed. The original Stage9C output folder is read only.
`--verify-preservation` verifies the saved before-review baseline while allowing
only the two explicitly recorded project-pointer documentation updates.
