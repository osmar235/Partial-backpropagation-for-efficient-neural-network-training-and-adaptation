# Stage14C independent review — 2026-10-24

**Decision: small positive reset interaction; practical quality/time criterion not met across both seeds.**

## Fixed final accuracy and adaptation cost

| Seed | Computation / optimizer | Final accuracy | Complete adaptation (s) |
|---|---|---:|---:|
| 1501 | full_restore | 84.72% | 30.507 |
| 1501 | spatial_fresh | 83.46% | 23.598 |
| 1501 | similarity_restore | 83.92% | 23.368 |
| 1501 | full_fresh | 84.70% | 29.413 |
| 1501 | spatial_restore | 83.42% | 23.921 |
| 1501 | similarity_fresh | 83.98% | 23.534 |
| 1502 | similarity_fresh | 83.76% | 22.957 |
| 1502 | spatial_restore | 82.92% | 22.619 |
| 1502 | full_fresh | 83.88% | 28.756 |
| 1502 | similarity_restore | 83.64% | 23.453 |
| 1502 | spatial_fresh | 82.82% | 23.315 |
| 1502 | full_restore | 83.90% | 28.287 |

## Reset effects and paired prediction changes

| Seed | Computation | Reset effect (pp) | Corrected | Regressed | Net correct / 5,000 |
|---|---|---:|---:|---:|---:|
| 1501 | full | -0.02 | 7 | 8 | -1 |
| 1501 | spatial | +0.04 | 5 | 3 | +2 |
| 1501 | similarity | +0.06 | 11 | 8 | +3 |
| 1502 | full | -0.02 | 5 | 6 | -1 |
| 1502 | spatial | -0.10 | 0 | 5 | -5 |
| 1502 | similarity | +0.12 | 14 | 8 | +6 |

A corrected example is wrong with restored history and correct with fresh history; a regressed example is the reverse. Changed predictions also include wrong-to-wrong changes. These counts are paired observations on reused development data, not independent training replications.

## Interpretation and decision

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


## Secondary diagnostic context

Similarity first-step gradients have identical aggregate norms within each history pair, while fresh-state updates have larger L2 norms (about 0.000931 versus 0.000201 / 0.000229). At the last step, fresh second-moment L2 norms are about 0.00181 / 0.00177 versus restored 0.00392 / 0.00398. Thus the optimizer intervention was active. Reset changes moments and bias-correction counters together; these diagnostics do not isolate a second-moment cause or show that larger updates are better. The endpoints govern the decision.

## Verification

All 2,179 protected earlier and raw files were hash-verified before and after review. No previous spec, source, result or manuscript was overwritten.

```json
{
  "status": "PASS",
  "training_updates_verified": 10560,
  "discarded_warmup_updates": 36,
  "independent_curve_points": 72,
  "saved_model_scoring_conditions": 18,
  "max_abs_rescored_logit_error": 0.0,
  "both_seed_reference_qualified": true,
  "both_seed_mechanism_direction": true,
  "both_seed_practical_pass": false,
  "historical_restored_trajectory_reproduced": true,
  "retained_arms_share_one_launch": true,
  "decision": "mechanism_only_no_efficiency_claim",
  "prior_and_raw_files_preserved": 2179,
  "parameter_updates_performed": false,
  "test_scoring_performed": false,
  "development_scoring_performed": true
}
```

![Fixed recovery trajectories](optimizer_history_trajectories.png)

Artifacts: independent_decisions.csv, paired_reset_effects.csv, independent_curves.csv, saved_state_rescore.csv, frozen_audit_recheck.json and independent_audit.json.
