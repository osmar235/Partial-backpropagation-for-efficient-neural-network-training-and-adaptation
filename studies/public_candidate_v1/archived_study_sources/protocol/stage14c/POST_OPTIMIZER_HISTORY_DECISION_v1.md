# Stage14C post-run decision — 2026-10-24

Stage14C is complete and independently reviewed. Fresh AdamW improves the
similarity-merging endpoint by +0.06 / +0.12 percentage points, while the matched
full model changes by -0.02 / -0.02 points. The reset-by-computation interaction
is therefore +0.08 / +0.14 points. Both seeds meet the predeclared directional
mechanism check, which required a positive gain and interaction, without a
minimum practical effect size. This is a small descriptive optimizer-state
sensitivity result, not proof of harmful second-moment memory or a new method.

Similarity fresh reaches 83.98% / 83.76%, versus best full 84.72% / 83.90%.
Remaining deficits are 0.74 / 0.14 points: seed 1501 fails the frozen 0.30-point
quality limit, seed 1502 passes. Both full references qualify; this is not
another broken-reference experiment. Spatial reset effects are mixed (+0.04 /
-0.10 points), with final fresh accuracy 83.46% / 82.82%; neither spatial
endpoint is quality-qualified. The primary two-seed practical screen FAILS.

Similarity reset corrects 11 and regresses 8 development predictions in seed
1501 (net +3/5,000), and corrects 14 and regresses 8 in seed 1502 (net +6/5,000).
Cross-entropy improves only slightly, by 0.0000129 / 0.0002640. Full reset loses
one net correct prediction per seed. These observations support reporting the
direction but not claiming reliable population improvement or significance.
The two source states and the same development examples were already examined.

Complete instrumented adaptation cost is 23.534 / 22.957 seconds, versus the
fastest required full comparator's 29.413 / 28.287 seconds. Savings are 19.99% /
18.84%, or 5.879 / 5.330 seconds. This passes the time requirement but cannot
substitute for quality. Adding the common source training cost reduces savings
to 1.08% / 1.00%; neither denominator establishes a matched-quality advantage.
Incremental training allocation is 26.39% lower and absolute training peak
18.48% lower. These are secondary measurements, not an alternative success rule.

The clock includes allocated shared data preparation, setup, warmup/restoration,
training, validation, diagnostics, training logs, checkpoints and artifact hashes.
It excludes post-endpoint inference, final summary/completion bookkeeping and
offline audit. It is an instrumented marginal adaptation clock, not standalone
cold-start latency. All retained arms share a single uninterrupted launch.

Independent review verified 10,560 retained updates, 36 discarded warmup updates,
72 saved score points and all frozen decisions. Six unique source computations
and twelve final models were rescored without updates; all logits reproduce
exactly. Paired initial logits are identical. Restored trajectories pass the
mandatory Stage14B reproduction checks. All protected prior/raw files remain
unchanged; no training or CIFAR test scoring occurred during review.

Decision: retain this small directional result and the failed practical screen
in the paper memory. Park fresh-optimizer token recovery as a practical candidate
under this fixed recipe. Do not expand to confirmatory seeds, extend epochs,
select an earlier epoch, relax the margin or run a reset/LR/boundary search on
these results. Reset alone has not solved the quality problem. Return to a
distinct mechanism/design decision before more training; do not infer which
alternative mechanism caused the remaining deficit. No next experiment is
prepared or launched. Publication remains on hold.
