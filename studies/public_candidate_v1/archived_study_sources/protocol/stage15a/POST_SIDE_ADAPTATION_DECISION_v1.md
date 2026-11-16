# Stage15A post-run decision — 2026-10-25

Stage15A is complete and independently reviewed. The multi-depth side candidate
FAILS the frozen quality requirement in both seeds despite passing the complete
adaptation-time requirement. Final accuracy is 82.60% / 82.68%, against strongest
control endpoints 84.72% / 83.88%. Deficits 2.12 / 1.20 percentage points exceed
the allowed 0.30 points. Both full references qualify and reproduce Stage14C
full_fresh trajectories. The reference is functioning; the tested side recipe
does not achieve enough adaptation benefit.

Multi-depth accuracy changes only +0.14 / -0.06 points from the original sources
(82.46% / 82.74%). Head-only ends at 82.66% / 82.74%; multi-depth is 0.06 points
lower in both seeds, or three fewer correct predictions per 5,000 examples.
Relative to head-only it corrects 27 / 13 predictions and regresses 30 / 16.
These small paired differences on reused development examples do not establish
statistical superiority or inferiority. Its cross-entropy is lower than head-only
by 0.00784 / 0.00717, so there is some loss improvement without sufficient accuracy.

The matched final-feature side control reaches 82.88% / 82.60%. Multi-depth minus
final-side is -0.28 / +0.08 points: no consistent accuracy advantage from the
intermediate taps. Neither side endpoint is quality-qualified. The intervention
was active: upstream side gradients activated after the zero-output first step,
and frozen backbones and optimizer histories passed the audit. This fixed recipe
does not identify whether capacity, optimizer settings, initialization or task
suitability limits learning. It does not refute Side-Tuning, LST or LAST, nor
constitute an exact reproduction or a new algorithm.

Complete instrumented adaptation is 21.100 / 21.647 seconds for multi-depth,
versus the fastest qualified control, head-then-full, at 28.588 / 27.299 seconds.
Savings are 26.19% / 20.71%, or 7.488 / 5.652 seconds. They are real measured cost
reductions at lower quality, not a matched-quality speedup. Including original
source training reduces those savings to 1.38% / 1.06%, still at lower quality.
The incremental training memory allocation is 83.12% lower and absolute training
peak allocation 59.04% lower than the required comparator. Inference costs are
1.327x / 1.451x the full model's measured batch time. These secondary resource
results do not replace the primary quality/time criterion; inference repetition
and the two reused source states do not establish population-level effects.

Head-then-full is the strongest inexpensive ordinary control: 84.72% / 83.80%,
within the quality margin in both seeds. Its changes versus full (84.70% / 83.88%)
are +0.02 / -0.08 points. Complete-time savings versus full are 4.28% / 8.54%,
below the predefined 10% threshold in both seeds. Preserve this useful descriptive
tradeoff without promoting it to the proposed method's success or a novel result.

All ten arms share one uninterrupted launch. The adaptation clock includes data
allocation, source/model setup, warmup/restoration, training, validation,
transition/gradient/state checks, logs, checkpoint/prediction saving and hashing.
It excludes post-endpoint inference, offline audit, startup integrity checks and
final summary/completion bookkeeping; it is not complete process latency.

Independent review verified 8,800 retained updates, 30 discarded warmup updates,
60 saved scoring points and all decision arithmetic. All ten source-condition
models and ten final models were scored without updates; all saved logits match
exactly. The review changed no earlier raw result, frozen source/spec, master,
memory or manuscript. It performed no training and never scored the CIFAR test
set. Development examples and source seeds remain reused exploratory evidence.

Decision: close this fixed side-network recipe. Preserve its substantial resource
savings and accuracy shortfall together in the paper memory. Do not add seeds,
extend epochs, select a better intermediate epoch, relax the margin or silently
sweep widths/taps/LR. The tested frozen-backbone alternatives did not capture
the benefit seen when adapting the backbone; the cause is not isolated. Return
to a distinct mechanism/design decision before further training. No next study
is prepared or launched. Publication remains on hold.
