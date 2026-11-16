# Next design: preserve the backbone, learn a small correction — 2026-10-25

## Selected direction

The author agreed to continue after Stage14C. Select **residual side-network
adaptation** for the next design step. The accepted objective remains lower
complete marginal adaptation cost at predefined quality, with inference secondary.
Stage14C's small reset gains and failed practical screen remain unchanged.

Question: can a compact trainable network use intermediate frozen representations
to recover most of the benefit of full adaptation, without backpropagating through
the backbone? This preserves all original input tokens and backbone computations.
It gives up changing the backbone itself. Neither sufficient capacity nor speed
is guaranteed by fewer trainable parameters.

This selection is a change of mechanism, not another reset, merge boundary,
learning-rate or epoch extension. It does not assert that representation loss
caused Stage14C's remaining deficit. We have not isolated that cause.

## Why this direction comes next

| Option | What it tests | Decision now |
|---|---|---|
| More optimizer-reset tuning | A stronger effect of the last intervention | Park: fixed recipe produced small gains and still failed quality |
| Less compression or another merge location | A different cost/quality operating point | Defer: would be a fresh search on already reused evidence, not the requested distinct mechanism |
| Teacher-guided compressed recovery | Whether extra teacher supervision repairs compression | Defer: teacher inference/storage must be charged and can consume the observed time saving |
| Approximate backward with full forward | Whether selected/approximated gradients preserve learning | Plausible reserve direction; custom backward work and actual timing need separate validation |
| Frozen full backbone with a residual side network | Whether a small global-loss-trained correction has sufficient adaptation capacity | Select for bounded feasibility and matched-control testing |

The Stage12 head-warmup/full control is a useful precedent for inexpensive
adaptation, but its ResNet/CIFAR-96 numbers cannot be transferred to this ViT.
Stage13 local objectives failed; the proposed side network receives the **final
task loss through one connected side computation**, rather than separately
trained detached local heads. This distinction is architectural, not evidence
that the new approach will work.

## Prior work: baseline first, no novelty claim

[Side-Tuning](https://arxiv.org/abs/1912.13503) already describes combining an
unchanged pretrained network with a learned additive side network.
[Ladder Side-Tuning](https://arxiv.org/abs/2206.06522) uses intermediate backbone
activations in a separate trainable network, avoiding backbone backpropagation.
[LAST](https://arxiv.org/abs/2402.04009) is a directly relevant vision adaptation
method using a side network with low-rank attention. Their results motivate
testing the family; they do not predict performance on our small ViT or hardware.

[TokenTune](https://aclanthology.org/2024.emnlp-main.1202/) is relevant to the
alternative of reducing backward work through selected tokens. It is not
evidence that our proposed side architecture is new or better.

The prototype below is inspired by this literature. It is **not an exact
reproduction of LST or LAST**, which have their own designs and recipes. Its
success would establish a useful local baseline or capacity result, not a new
algorithm. A future contribution must demonstrate a specific additional benefit
against strong published methods on a shared workload. No exhaustive novelty
search or competitive benchmark has been completed.

## Concrete prototype and completed structural proof

Use the existing six-block width-192 ViT shape, retaining all 65 tokens. Freeze
the backbone; copy its classifier as a trainable head. Read detached feature maps
after blocks 2, 4 and 6. At each tap, normalize/project to width 48 and process
with a three-head, width-48 side transformer block (MLP width 192, no side dropout).
The next side block receives its preceding side state plus the next projected
feature map. No side activation is inserted back into the frozen backbone.

Project the final side CLS representation back to width 192 using a zero-initialized
output matrix. The classifier combines the original representation and correction:

`logits = head(original_features) + head.weight @ side_correction`

The bias appears only once. Initially the correction is zero. Backbone dropout
follows train/eval mode and is kept identical across prospective controls. The
entire backbone forward occurs before side computation; side RNG consumption
therefore cannot perturb its current-pass dropout. Frozen parameters do not
automatically imply eval mode. No BatchNorm exists in this ViT.

The artificial CPU/CUDA proof checks both train-mode and eval-mode initial
outputs, zero backbone gradients, unchanged backbone hashes after updates,
active side learning and optimizer state round-trip reproduction. Both passed.
The prototype has 125,002 trainable and 2,691,648 frozen parameters. These counts
do not measure wall-time savings. It adds inference work to the full backbone;
an inference acceleration is not expected from this construction.

Initial outputs are bitwise identical to the native **frozen** source control.
Relative to the native unfrozen source, the largest artificial logit discrepancy
is about 5.22e-7; predicted classes match. Investigation showed freezing alone
reproduces this difference. The proof therefore distinguishes exact agreement
with the frozen control from the existing 1e-5 numerical tolerance against the
unfrozen source. It does not conceal the numerical execution difference.
Real preparation must verify starting classes/metrics too, not assume that a
small logit tolerance rules out changes on near-tied examples.

Zero output initialization gives zero upstream side gradients on the first step;
the output matrix and head learn first. Upstream side gradients become active
on the second step. This is tested explicitly, not silently treated as a missing
gradient bug. The proof uses only artificial inputs, labels and model weights.
No real source checkpoint was loaded and no task accuracy or speed was measured.

Files: `scripts/side_adaptation_prototype.py`, `scripts/prove_side_adaptation.py`,
and `results/side_adaptation_design_v1/`. A reporting-only variable-shadowing bug
in the first proof output mislabeled both arms; raw outputs are preserved and
the correction is recorded explicitly. Numeric checks were unaffected.

## Proposed matched screen: five arms, one candidate

Use the same two original Stage11B full checkpoints as starting sources, not
the Stage14C adapted endpoints. This remains same-task recovery on reused
development states, not downstream transfer or confirmation. Keep five epochs,
the existing data/MixUp/augmentation schedule and fresh AdamW in all arms as a
starting design. Source optimizer history is no longer an experimental variable.

| Arm | Trainable component | Purpose |
|---|---|---|
| Full adaptation | Entire native model | Matched quality/cost reference |
| Head only | Copied classifier | Cheapest ordinary feature-extractor control |
| Head warmup, then full | Classifier for epoch 1; full model for epochs 2–5 | Re-test the useful ordinary strategy on this exact workload |
| Final-feature side control | Same side architecture and parameter count, each tap reading the final feature map | Controls added capacity/compute without multi-depth access |
| Multi-depth side candidate | Side network reads blocks 2/4/6, plus classifier | Tests whether intermediate access matters |

Both side variants have identical parameter initialization and counts. The
final-feature control is intentionally stronger than a small linear head;
without it, a side-network gain could just reflect a larger nonlinear readout.
Ten trajectories would result, not a width/depth/LR sweep. The complete real
protocol, fresh-parameter initialization, phase-transition optimizer handling,
orders, source/environment checks and auditor remain to be finalized and frozen.
This document is not a registered runnable real experiment.

## Decision gates before spending another real training run

1. Structural invariants on artificial CPU/CUDA: completed. Preserve all source
   tokens and starting predictions within the explicitly checked numerical rules;
   ensure no hidden backbone gradient path and no side feedback into it.
2. **Cost feasibility next:** benchmark full forward/loss/backward/AdamW against
   head-only and both side variants on the target device, with full batch shapes,
   discarded warmup, fresh-state allocation and both steady-state and cold setup
   reported. Interleave repeated timing trials; retain raw measurements. Include
   all backbone forwards and feature-capture/side overhead, without feature caches
   that would silently reuse different stochastic augmentations. A side route
   that cannot save at least 10% even at this operation level does not justify
   this fixed time-efficiency screen. A pass only permits preparation; it does
   not establish complete-run speed or quality. Never choose widths by a sweep
   after seeing timings in this first bounded proposal.
3. If cost is viable, finalize/test/freeze the ten-run real screen and then give
   the author CMD lines. Do not run real training automatically. Count setup,
   data work, warmup, training, validation, logging and checkpoint costs. Preserve
   separate inference costs; full-forward side networks may increase them.
4. Proposed practical criterion: candidate within 0.30 percentage points of the
   strongest relevant ordinary/side-control final endpoint and original full
   source, in both seeds; at least 10% lower complete adaptation time versus full
   and the fastest quality-qualified ordinary control. The final-feature side
   control also belongs in that ordinary-control set. Reference full must meet
   its frozen readiness requirement. These thresholds require operationalization
   before real results, not reinterpretation of past failures.

If the final-feature control is equally useful, prefer the simpler explanation;
do not attribute success to intermediate information. If neither side variant
meets quality, close this fixed-capacity recipe without declaring all side-tuning
ineffective. If a valid practical advantage appears, independent training seeds,
an external-checkpoint transfer workload, published baselines and a clear novelty
case would still be needed. No confirmatory launch follows automatically.

## Current status

Direction selected; structural prototype validated; cost feasibility remains
unmeasured; real study is not frozen and no new real-run command is released.
This is concrete design progress, not a new efficacy result. Earlier artifacts
and failed screens remain protected. Publication remains on hold.
