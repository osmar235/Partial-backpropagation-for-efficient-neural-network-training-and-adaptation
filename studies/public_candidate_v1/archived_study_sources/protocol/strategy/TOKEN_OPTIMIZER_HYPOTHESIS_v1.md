# Step 2: candidate and hypothesis — 2026-10-24

## Decision and scope

The author accepted checkpoint adaptation as the primary objective, with
inference cost secondary. Select **similarity-based token merging** for the
next bounded mechanism diagnostic. The one unresolved hypothesis is:

> Keeping AdamW history learned with full tokens may hinder recovery after
> switching to merged tokens. Starting with fresh optimizer state may reduce
> the accuracy deficit more for the merged model than for an equally treated
> full-token model, while retaining a useful adaptation-time saving.

This selects a candidate and a falsifiable question. It does not establish
the mechanism, endorse generic resetting, or authorize promotion of Stage14B.
The design below is a proposal for protocol preparation, not a frozen runnable
study. No experimental code, training command or new real outcome is produced
by this selection step.

## Why this candidate and this variable

Stage14B already demonstrates 22.55% / 21.51% lower charged recovery time,
26.39% lower incremental training allocation and an inference-operation
advantage. Final deficits of 0.80 / 0.26 points show that reliable quality is
the unresolved practical issue. Source AdamW history at age 17,600 was
deliberately restored in every arm; its compatibility with the changed
computation was never varied in that experiment.

The frozen restore function checks betas (0.9, 0.999). For the Adam recurrence
v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2, the coefficient on the starting
second-moment estimate after k updates is beta2^k. At k=880 this is about
0.415. The corresponding first-moment coefficient, beta1^880, is negligible.
These are exact recurrence coefficients under those betas, **not** the
fraction of the final moment attributable to the source, a measure of harmful
staleness, or evidence that Adam cannot adapt. The recent-gradient term can
be large or small. Resetting also changes effective update scaling and bias
correction, so a reset effect would not by itself isolate a second-moment cause.
The definition of Adam's moment estimates is established in the
[Adam paper](https://arxiv.org/abs/1412.6980).

Compression changes the gradient computation even though parameter shapes
stay the same. Our hypothesis concerns recovery after that switch. It cannot
explain Stage14A's immediate compression loss, which occurs before optimizer
updates. Irrecoverable information loss, the merge rule, regularization and
the fixed learning-rate recipe remain alternative explanations.

Stage13B is a caution against assuming reset helps: its different ResNet
continuation experiment found mixed endpoint effects and worse first-step
quality after reset. It did not test this ViT/token-compression interaction.
Use a fixed multi-step endpoint; never accept/reject on one-step positivity.

## Proposed minimal comparison

Use the same source checkpoints, merge boundary, token count and five-epoch
budget as Stage14B, with a matched 3-by-2 design:

| Computation | Restore source AdamW history | Fresh AdamW state |
|---|---|---|
| Full tokens | Full continuation control | Full reset control |
| Spatial merging | Simple compressed control | Simple compressed reset control |
| Similarity merging | Stage14B candidate recipe | Primary candidate for this diagnostic |

At two source states this is twelve trajectories, 880 updates each, 10,560
retained updates. Those states and the development examples have already been
examined; the experiment would be mechanism exploration, not confirmation.
All six arms must be rerun under the same execution and timing rules. Historical
Stage14B results are provenance/reference evidence, not free contemporary
timing controls. Use the old matched streams as reproduction checks for the
restored arms if the final protocol adopts them; any change must be declared.

Only optimizer-state initialization changes: identical source weights,
65 -> 33 tokens after block 3, data/augmentation/MixUp streams, dropout seeds,
five-epoch schedule, base learning rate, warmup, weight decay and endpoint.
Fresh means reset both moments AND the corresponding step/bias-correction
state for all parameters. Do not zero moments while leaving mature counters,
reset weights, silently restart a different LR schedule or introduce a
selective-layer reset in this first comparison. All parameters' gradients can
change through merging, including parameters before the merge boundary.

Restored versus fresh state must produce identical initial logits within
each computation mode. Hardware warmup is discarded, then the assigned
weight/optimizer condition restored exactly. First-state allocation, setup,
warmup, restoration and training costs must be charged; report peak allocated
memory with persistent optimizer-state storage visible. Include evaluation,
logging and checkpoint costs in the complete marginal adaptation clock and
retain scoped training times separately. Matched handling of fresh optimizer
allocations is required before resource interpretation.

## Two different questions to answer

**Mechanism question:** for each seed, calculate the reset interaction in
accuracy points:

I = (similarity_fresh - similarity_restored)
    - (full_fresh - full_restored).

Positive I means the deficit relative to the correspondingly treated full
model narrows. A compressed model improving alone does not establish that
interaction. Report the spatial interaction, cross-entropy, trajectories,
update magnitudes and first/second-moment diagnostics as secondary evidence.
Two positive observations would be descriptive support, not significance or
a demonstration that second-moment memory alone is responsible.

**Practical question:** does the final similarity/fresh candidate retain
quality relative to the strongest relevant control while reducing complete
marginal adaptation time? Keep a proposed 0.30-point quality margin, evaluated
in each seed, and compare with the original full source plus all full/spatial
control endpoints. An optimizer change must not win only because its full
comparator deteriorated. For a time-efficiency claim, propose at least 10%
complete adaptation-time saving versus full and the fastest quality-qualified
ordinary control; a memory saving is a separate outcome, not a substitute for
that primary time claim. If spatial is equally good and cheaper, prefer it.

These are design commitments to operationalize and freeze before execution,
not a new analysis of the old runs. The final protocol must specify reference
qualification, exact clocks, orders, numerical tolerances, scoring steps,
mechanism-support interpretation and stop rules before real data are generated.

## How the outcome would change the project

- Positive interaction plus a practical win: supports a larger investigation
  of optimizer-state compatibility under compression; it is not yet confirmation.
- Reset improves full and merged models similarly: a general recipe effect,
  with no evidence of the proposed compression-specific benefit.
- Interaction is favorable but quality or time still fails: a mechanistic
  observation, not a useful efficiency result under the declared requirement.
- Mixed or negative interaction: no support across both tested states; close
  this hypothesis without automatically trying more reset locations or epochs.
- A practical win without the predicted interaction: preserve the practical
  result but reject this mechanism as its explanation.

Do not select the best epoch, relax old margins, repeat until favorable seeds
appear or launch a broad reset/LR/compression search. A failed recipe does not
disprove compression generally. After a promising bounded diagnostic, freeze
the selected method and plan fresh training replication plus a suitable
independent evaluation before a broader claim.

## Prior work and limits of novelty

[Token Merging (ToMe)](https://arxiv.org/abs/2210.09461) already studies token
merging during inference and training, including fine-tuning. Our fixed
residual-feature merger is ToMe-inspired, not an exact reproduction. A positive
result for ordinary merging with a fresh optimizer is not itself a new algorithm.
It could simply expose a limitation of our previous continuation recipe.

[LP-FT](https://arxiv.org/abs/2202.10054) establishes head-first then full
fine-tuning as prior work. Its role here is a strong ordinary adaptation
baseline to evaluate on a common workload before any final competitive claim;
the ResNet result cannot serve as a directly comparable ViT baseline.
The minimal diagnostic above tests a specific within-model mechanism and is
not an exhaustive adaptation benchmark.

The focused primary-source check supports those background facts and Adam's
moment definition. It does not establish that the proposed interaction is
novel, or that no existing paper has examined it. No exhaustive novelty search
has been completed.

This first diagnostic concerns same-task recovery from checkpoints that retain
optimizer history. If an external checkpoint provides only weights, there is
no source Adam history to reset or restore; the retained-history comparison
does not directly apply. Broader transfer claims require a separate common
workload with suitable external checkpoints, strong baselines and fresh
evaluation. Keep this limitation explicit in any eventual paper.

## Provenance

Source evidence: results/stage14B_review_v1/REPORT.md,
results/research_objective_comparison_v1/REPORT.md,
results/stage13B_review_v1/REPORT.md, scripts/stage14_recovery.py and
protocol/stage14b/recovery_spec_v1.json. All remain unchanged. The numerical
moment-memory calculation is saved separately with its assumptions. No new
training, predictions, held-out scoring or frozen experimental sources were
changed in this selection step. Publication remains on hold.
