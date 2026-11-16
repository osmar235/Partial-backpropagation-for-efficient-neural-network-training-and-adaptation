# AGRLS Stage 5C — exact blockwise Adam

Stage 5B showed that a four-direction history cannot reconstruct a useful full
gradient. Stage 5C keeps every gradient exact and saves work by differentiating
only a trainable suffix on selected updates.

## Stage 5C0 hardware calibration

The calibration used CIFAR-10, ResNet-18 with GroupNorm, batch 256, and 20 timed
repetitions on the RTX 5080.

| trainable suffix | parameter coverage | backward | forward + backward | saving vs full |
|---|---:|---:|---:|---:|
| classifier | 0.05% | 0.522 ms | 13.575 ms | 66.23% |
| layer4 + classifier | 75.16% | 4.674 ms | 17.798 ms | 55.73% |
| layers3–4 + classifier | 93.96% | 8.255 ms | 21.262 ms | 47.11% |
| layers2–4 + classifier | 98.66% | 14.766 ms | 27.748 ms | 30.98% |
| full model | 100% | 27.145 ms | 40.204 ms | — |

`layer4 + classifier` is the frozen first candidate. It covers three quarters
of the parameters and has enough timing margin that alternating it with a full
update has an estimated forward/backward saving near 28%.

## Frozen Stage 5C1 test

- CIFAR-10 development seed 0 only
- 500 updates, batch 256, Adam learning rate 0.001
- odd updates: exact full-model Adam gradient
- even updates: exact `layer4 + classifier` Adam gradient
- no probes, estimated gradients, or copied displacements
- PyTorch Adam uses its ordinary per-parameter step counters; frozen parameters
  do not advance their moments on partial updates

Proceed only if evaluation accuracy is within 1 percentage point of the paired
Adam baseline, measured training time falls by more than 10%, and wall time is
lower. Untouched seeds remain locked until those criteria pass.

## Run

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage5c0_cifar10_block_backward_calibration.cmd
run_stage5c1_cifar10_alternating_block_adam_seed0.cmd
```

## Stage 5C1 result

Stage 5C1 reduced measured training time by 27.54% and wall time by 27.27%, so
the blockwise implementation produced a genuine end-to-end speedup. Evaluation
accuracy was 51.82% versus 58.70% for paired Adam, a deficit of 6.88 percentage
points. The frozen accuracy criterion failed.

## Frozen Stage 5C2 conservative test

Stage 5C2 changes the partial update to `layers2–4 + classifier`. It computes
exact gradients for 98.66% of parameters and omits only the stem and layer 1 on
even updates. The hardware calibration measured 30.98% savings for each partial
update, giving an expected alternating saving near 15%. The criterion remains
unchanged: within 1 evaluation-accuracy point of paired Adam, more than 10%
measured training-time reduction, and positive wall-time reduction.

```cmd
run_stage5c2_cifar10_alternating_layer2_4_adam_seed0.cmd
```

## Stage 5C2 result

Stage 5C2 reduced measured training time by 14.84% and wall time by 14.70%.
Evaluation accuracy was 57.62% versus 58.70% for paired Adam, a deficit of 1.08
percentage points. It missed the frozen accuracy tolerance by 0.08 point and is
therefore recorded as a failure rather than rounded into a pass.

## Frozen Stage 5C3 schedule

The final seed-0 schedule uses the same exact `layers2–4 + classifier` partial
gradient on two of every five updates. The repeating pattern is full, full,
partial, full, partial. This gives the stem and layer 1 exactly 300 gradients
instead of Stage 5C2's 250, while the Stage 5C0 timing predicts approximately
12% savings. The pass criteria remain unchanged.

```cmd
run_stage5c3_cifar10_block_adam_two_of_five_seed0.cmd
```

## Stage 5C3 result

Stage 5C3 reduced measured training time by 14.21% and wall time by 14.29%, but
evaluation accuracy was 55.10% versus 58.70% for Adam, a 3.60-point deficit.
The non-monotonic accuracy response shows that which batches update the early
block matters; no further update-frequency tuning is allowed on seed 0.

## Stage 5C4 equal-wall test

Stage 5C2's observed wall ratio permits approximately 586 alternating updates
inside the time used by 500 Adam updates. The horizon is conservatively frozen
at 580. Stage 5C4 reuses the unchanged Stage 5C2 half-partial mechanism. It
passes only if its accuracy is within 1 point of Adam and both measured training
time and wall time do not exceed the paired Adam run.

```cmd
run_stage5c4_cifar10_block_adam_equal_wall_seed0.cmd
```

## Stage 5C4 result and Stage 5C5 freeze

At essentially equal wall time, the frozen 580-update blockwise candidate
reached 61.50% evaluation accuracy versus 58.70% for 500 Adam updates. Measured
training time was 1.20% lower and wall time was 1.15% lower. Stage 5C4 therefore
passed its development criterion.

Freeze the exact mechanism and horizons for held-out CIFAR-10 seeds 1–9. Odd
seeds run Adam first and even seeds run the candidate first. Do not inspect an
individual seed to change the algorithm; aggregate all nine before making a
decision. The aggregate passes only if median accuracy delta is at least -1
point, median measured and wall-time reductions are nonnegative, and at least
6/9 seeds separately meet each of those three conditions.

```cmd
run_stage5c5_cifar10_heldout_seeds1_9.cmd
```

## Stage 5C5 held-out result

The frozen batch completed all nine CIFAR-10 seeds without a technical failure.
No algorithm or horizon was changed after the batch began.

| aggregate metric | result | required |
|---|---:|---:|
| median Adam accuracy | 59.95% | reference |
| median blockwise accuracy | 58.00% | — |
| median paired accuracy delta | -0.47 pp | >=-1.00 pp |
| median measured-time reduction | +0.67% | >=0% |
| median wall-time reduction | +0.67% | >=0% |
| seeds within 1 accuracy point | 5/9 | >=6/9 |
| seeds with noninferior measured time | 6/9 | >=6/9 |
| seeds with noninferior wall time | 5/9 | >=6/9 |

The aggregate therefore **failed** the frozen criterion. The medians passed all
three central thresholds, but the two 6/9 robustness counts did not. Seed 0's
large +2.80-point development gain did not reproduce reliably: held-out paired
accuracy deltas ranged from -4.47 to +0.93 points.

The systems result is also narrower than the Stage 5C0 calibration suggested.
At 580 candidate versus 500 Adam updates, the median end-to-end saving was only
0.67%, and execution-order counterbalancing exposed several runs in which the
candidate was slightly slower. This is a competitive equal-wall operating
point, not robust evidence that the candidate is faster than Adam.

## Decision

Do not claim an optimizer improvement from Stage 5C. Do not tune the block,
schedule, or 580-update horizon on seeds 1–9; they are now seen. The branch did
establish two useful facts:

1. exact suffix gradients can reduce a 500-update run by about 15% on the
   development trajectory while remaining close to Adam's accuracy;
2. spending that saving on more updates produces median-held-out accuracy within
   1 point of Adam, but the accuracy and wall-time effects are not yet robust.

Any next algorithm revision needs a new development setting and a new untouched
seed block. A credible speed claim should target more than the present 0.67%
median margin so ordinary GPU timing variation cannot reverse the conclusion.
