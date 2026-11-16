# Stage12A completed review — 2026-10-22

Periodic pretrained-prefix updates fail the frozen final-quality criterion in
both seeds. Baseline readiness and speed pass, so this is an interpretable
quality failure for the tested recipe. Preserve the observed tradeoffs and the
useful head-warmup control; do not equate integrity PASS with efficacy PASS.

| Seed | Full accuracy | Periodic accuracy | Difference | Full complete time | Periodic complete time | Time saving |
|---|---:|---:|---:|---:|---:|---:|
| 1601 | 94.76% | 93.24% | -1.52 pp | 139.75 s | 83.97 s | 39.91% |
| 1602 | 94.56% | 93.60% | -0.96 pp | 137.19 s | 83.94 s | 38.81% |

The fixed rule allowed at most0.30percentage points below the best control and
required at least10% complete-run time saving versus BOTH full and the fastest
quality-qualified control, in BOTH seeds. Full and head-warmup-then-full (lp_full)
are quality-qualified; lp_full is faster. Periodic saves31.98% /30.95% against
lp_full, but still loses1.26 /0.76points to that control. Its failure cannot be
explained by comparing only with the slower full-from-start baseline.

Both full-backbone controls sustain90% and92%; periodic also sustains both.
There is no baseline-readiness failure. All runs completed the fixed12epochs.

## All five arms

| Arm | Seed1601 final accuracy | Seed1602 final accuracy | Mean accuracy | Mean complete-run seconds |
|---|---:|---:|---:|---:|
| Full from start | 94.76% | 94.56% | 94.66% | 138.47 |
| Head only | 82.76% | 82.70% | 82.73% | 43.77 |
| Head warmup, then full | 94.50% | 94.36% | 94.43% | 122.51 |
| Head warmup, then fixed suffix | 91.54% | 91.40% | 91.47% | 70.84 |
| Head warmup, then periodic full | 93.24% | 93.60% | 93.42% | 83.96 |

Head-warmup-then-full loses only0.26 /0.20points relative to full, with11.66% /
11.38% less complete time. This ordinary control meets the declared quality
tolerance relative to full and illustrates a modest useful saving in this
setting. It does NOT become a replacement primary candidate after seeing the
results, and it establishes no new algorithm or statistically confirmed effect.

Periodic full updates improve on fixed suffix by1.70 /2.20points at18.63% /
18.39% extra time. The paths share bitwise-identical head warmups, data streams,
head/suffix update counts and LR schedule; the periodic arm adds880prefix
updates. This supports the value of some prefix adaptation in this setting,
while showing that this sparse schedule does not recover full-update quality.
It does not identify an optimal update rate or imply all selective-update
methods will fail. Head-only adaptation is insufficient under this recipe.

Mean results describe two seeds only and do not replace the per-seed decision.
The data support an empirical accuracy/time tradeoff, not a reliable solution
to the original near-full-quality efficiency objective.

## Secondary target results, not a new primary endpoint

Periodic reaches sustained92% at45.49 /45.60observed seconds versus
65.11 /64.27seconds for lp_full:30.14% /29.05% sooner. Fixed suffix and head
never sustain92% within12epochs. Periodic therefore has a useful observed
lower-target tradeoff even though final-quality acceptance fails.

These are the predeclared secondary target's post-hardware-warmup observed
clocks, distinct from the complete-run primary clock. They do not establish
that a92%-target deployment will reproduce the benefit, and they do not justify
changing the study's primary target or acceptance criterion after the fact.

## Verification

- Re-ran frozen read-only audit: ten runs,42,240updates,27source dependencies,
  official pretrained checkpoint, configuration/split, traces and saved states.
- Verified fixed BatchNorm buffers, unchanged never-trained parameters, paused
  optimizer semantics and exact active-group counters. Periodic counts per run:
  prefix880, suffix3520, head4224. lp_full has3520prefix updates; full has4224.
- Matched head-warmup weights, Adam states, losses and validation histories are
  bitwise identical. Data RNG streams match across all five arms.
- Independently reconstructed threshold crossings, censoring, qualified controls,
  final quality margins, speed ratios and both-seed decisions from raw records.
- Reloaded every final model and rescored the original5,000-example development
  validation split on CUDA. All ten accuracies reproduce exactly; losses match
  within1e-6. No training and no test-set evaluation occurred during review.
- Verified1,069immutable files before/after review, including prior results,
  sources, masters, preparation records, pretrained weights and real raw outputs.
  Review files are separate; only current pointers are intentionally updated.

## Limits and next decision

This is marginal adaptation from external ImageNet pretraining, with96px input,
fixed BatchNorm running statistics, deterministic FP32 and one12epoch recipe.
It is not a fully tuned224px/BN-updating baseline comparison, a pretraining-cost
claim, an independent test evaluation or an exhaustive study of freezing.
Validation has been reused for development; two seeds provide no significance
or generalization claim. The LP-FT and freezing baselines are established ideas,
as documented in the frozen protocol.

Close the fixed one-in-four prefix-update recipe under the original rule.
Keep the useful observations: partial prefix recovery, a lower-target speed
tradeoff, and a stronger inexpensive ordinary lp_full baseline. Retain the prior
growth evidence and its different limitations. No automatic LR/period/boundary
search or extra epochs; publication remains on hold.

The next queued idea is local objectives / splitting credit assignment into
smaller training problems. Its future protocol should start with a small
mechanics-and-cost diagnostic and compare against ordinary full backpropagation.
No new training was launched or new experiment command prepared in this review.
