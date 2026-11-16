# Stage 6: publication-protocol validation

This stage tests the Stage 5F mechanism under an augmented, epoch-based CIFAR-10
training protocol. The optimizer controller is frozen before inspecting Stage 6
outcomes.

## Frozen optimizer controller

- Candidate: `loss_gated_moment_continuation_adam`.
- Partial update gate: current augmented-batch loss is no greater than the
  pre-update loss EMA.
- EMA decay: 0.95.
- A full update is forced after every partial update.
- Prefix: stem plus `layer1`; suffix: `layer2`, `layer3`, `layer4`, and `fc`.
- A partial update computes exact suffix gradients and supplies explicit zero
  prefix gradients to Adam, advancing the prefix moment state and bias step.

## Training protocol

- CIFAR-10 with all 50,000 training and 10,000 test images.
- ResNet-18 with GroupNorm, unchanged from Stage 5.
- Random crop to 32x32 after four-pixel zero padding, random horizontal flip
  with probability 0.5, then the fixed CIFAR-10 normalization.
- Batch size 256, including the final incomplete batch of each epoch.
- Per-update cosine learning-rate decay to 1% of the initial rate.
- Adam and the candidate use the frozen Stage 5 learning rate of 0.0003. The
  official Adan baseline also uses its frozen Stage 5 learning rate of 0.0003.
- Algorithms receive identical model initialization, epoch permutations, crop
  offsets, and flip decisions within a seed.
- Validation runs after every epoch. Its cost is excluded from training wall
  time and reported separately.

## Staged evidence plan

- Stage 6A0: fake-data smoke test for correctness.
- Stage 6A1: one 30-epoch development run on seed 40. This may diagnose the
  protocol but cannot support a confirmatory claim.
- Seeds 41-49 remain sealed for the next confirmatory experiment. Any protocol
  change motivated by seed 40 must be written here before those seeds run.
- Predeclared time-to-target levels: 55%, 60%, 65%, 70%, 75%, 80%, and 85%
  test accuracy. A target is first reached at the end of an epoch; reported time
  is cumulative training time and excludes validation time.

## Development decision after seed 40

Proceed to held-out seeds only if the run completes without correctness issues,
the controller remains numerically stable, and the 30-epoch horizon reaches a
useful part of the learning curve. A longer horizon is permitted if none of the
methods has begun to plateau, but it must be selected before opening seeds 41-49.

## Seed-40 timing crossover decision

The first development run completed successfully. Adam reached 84.87% test
accuracy and the frozen candidate reached 84.81%, while the candidate used
36.41% partial updates. The candidate ran 5.31% longer, but its augmentation,
forward, and optimizer components were all slower even where the operations were
identical. Because the run order was Adam, Adan, candidate, this timing result is
confounded by order or GPU-state drift.

Before opening held-out seeds, repeat the identical 30-epoch seed-40 Adam versus
candidate comparison in reverse order: candidate first, Adam second. Do not
change the controller, learning rate, augmentation, schedule, initialization, or
random streams. Use the two orders together to decide whether end-to-end speed
is supported and to choose a balanced order for confirmatory seeds.

The crossover produced candidate/Adam wall-time ratios of 1.053 when the
candidate ran last and 0.911 when it ran first. The geometric order-balanced
ratio was 0.979, but the order interaction was too large to treat the roughly
2.1% estimate as reliable. Mean final candidate minus Adam accuracy across the
two runs was -0.325 percentage points.

Before confirmatory seeds, add one algorithm-specific warm-up epoch (196 steps)
immediately before each timed run. The warm-up uses a separate random stream,
is excluded from all reported times, and is followed by restoration of the
shared initial model and fresh optimizer state. This is timing instrumentation;
it does not change the controller or the timed training protocol. Validate this
instrumentation on short seed-40 order crossovers before opening seeds 41-49.

The two five-epoch timing qualifications used this warm-up and showed candidate
wall-time reductions of 11.72% with Adam first and 8.96% with the candidate
first. Their order-balanced geometric estimate was 10.35%. Use the same warm-up
for all later timing runs.

The official Adan repository states that its peak learning rate is generally
five to ten times the Adam/AdamW rate. The Stage 6A1 Adan value of 0.0003 is
therefore not an adequate tuned baseline. On development seed 40, screen Adan
learning rates 0.001, 0.0015, and 0.003 for the first 10 epochs of the unchanged
30-epoch cosine schedule. Select the highest epoch-10 test accuracy; if values
are within 0.25 percentage points, select the smaller learning rate. Do this
before opening seeds 41-49.

The screen completed with epoch-10 accuracies of 69.80% at 0.001, 66.84% at
0.0015, and 64.04% at 0.003. Freeze Adan at 0.001 for the 30-epoch protocol.
Run one full development curve at this setting before the held-out study.

The full tuned-Adan development run reached 82.12% test accuracy in 263.90
timed seconds. Freeze the confirmatory protocol as follows:

- Held-out seeds: 41-49, opened only by the final harness.
- Methods: Adam at 0.0003, Adan at 0.001, and the frozen candidate at 0.0003.
- Thirty epochs, a 30-epoch cosine horizon, and 196 excluded algorithm-specific
  warm-up steps before each timed run.
- Balance run position with a three-order Latin square repeated three times:
  Adam/candidate/Adan, candidate/Adan/Adam, and Adan/Adam/candidate.
- Primary accuracy estimand: mean paired candidate-minus-Adam final test
  accuracy. Non-inferiority requires the lower bound of a two-sided 95% t
  interval to exceed -1.0 percentage point.
- Primary timing estimand: geometric mean candidate/Adam training-wall ratio.
  A speed claim requires the upper bound of a two-sided 95% t interval on the
  log ratio to be below 1.0.
- Candidate versus tuned Adan and time-to-target results are secondary.
- Do not change this protocol after seed 41 is opened.
