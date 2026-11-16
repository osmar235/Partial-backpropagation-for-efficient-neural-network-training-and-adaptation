# Stage14B independent review — 2026-10-24

**Decision: park this fixed recovery recipe.** Integrity passes, and both full
references qualify. Neither compressed arm passes the frozen two-seed screen.
Similarity merging improves absolute accuracy and retains resource savings,
but its quality deficit against equally adapted full training persists in one
seed. No threshold change, best-epoch substitution or extension is warranted
by this result.

## Fixed final outcomes

Six runs completed five epochs each: 880 retained updates per arm, 5,280 total,
plus 18 discarded hardware-warmup updates. The same source weights and AdamW
histories were restored in all three arms per seed. All metrics below use the
predetermined fifth-epoch checkpoint and reused development examples.

| Seed | Arm | Initial accuracy | Final accuracy | Improvement | Final gap vs adapted full | Charged recovery |
|---|---|---:|---:|---:|---:|---:|
| 1501 | Full | 82.46% | 84.72% | +2.26 pp | — | 26.281 s |
| 1501 | Spatial | 81.32% | 83.42% | +2.10 pp | -1.30 pp | 20.288 s |
| 1501 | Similarity | 82.16% | 83.92% | +1.76 pp | -0.80 pp | 20.354 s |
| 1502 | Full | 82.74% | 83.90% | +1.16 pp | — | 26.586 s |
| 1502 | Spatial | 81.00% | 82.92% | +1.92 pp | -0.98 pp | 20.343 s |
| 1502 | Similarity | 82.14% | 83.64% | +1.50 pp | -0.26 pp | 20.868 s |

Full training is the best quality control in both seeds, and both references
remain above their initial quality and 80% readiness floor. Similarity passes
the inclusive 0.30-point quality margin in seed 1502, but fails in seed 1501.
Spatial fails quality in both. Both compressed modes pass both resource routes
in both seeds. The joint quality/resource rule still fails for each method.

Similarity beats the *original, unadapted* full sources by 1.46 / 0.90 points.
That alone would overstate success: equally fine-tuning full training improves
it by 2.26 / 1.16 points. Relative to the immediate Stage14A gap, similarity's
deficit widens from 0.30 to 0.80 points in seed 1501 and narrows from 0.60 to
0.26 points in seed 1502. Recovery is not consistent relative to the matched
control. All arms lower cross-entropy, but that does not replace the frozen
accuracy rule.

Similarity exceeds adapted spatial accuracy by 0.50 / 0.72 points at comparable
recovery cost. This is a descriptive advantage within this comparison. The
predeclared priority rule first requires a two-seed similarity screen pass;
that prerequisite fails. Do not promote similarity by citing only its advantage
over the weaker spatial control.

![Recovery trajectories](recovery_trajectories.png)

## Resources and the source-cost distinction

Similarity saves 22.55% / 21.51% of charged recovery time, including source/model
and optimizer setup, discarded warmup, exact restoration, data transformations
and actual forward/backward/AdamW updates. Unlike Stage14A's operation-only
diagnostic, these measurements include optimizer updates. Evaluation, logging,
telemetry, checkpoint saving and inference benchmarking remain outside that
scoped charge and are retained in the raw summary/observed arm costs.

Similarity reduces maximum incremental training-operation allocation by
26.39% in both seeds, and absolute training peak allocation by 18.48%. These
are different quantities; 26.39% is not total VRAM savings. Post-adaptation
inference operation time is 23.34% / 21.92% lower, with 15.38% lower incremental
inference allocation. Inference figures are secondary measurements, not a
separate confirmatory pass or a deployment-throughput guarantee.

The fully trained source cannot be treated as free for a training-efficiency
claim:

| Seed | Source training + full recovery | Source training + similarity recovery | Saving |
|---|---:|---:|---:|
| 1501 | 541.246 s | 535.319 s | 1.10% |
| 1502 | 532.552 s | 526.834 s | 1.07% |

Adding original complete-run cost to observed recovery-arm cost gives savings
of 1.15% / 1.10%. These are sums of recorded costs from separate executions,
not newly observed end-to-end timings. Common data loading took 0.841 s for
this launch and is reported separately; the source-inclusive totals do not
include the entire research program or unsuccessful experiments. The candidates
also do not achieve matched final quality in both seeds. Thus neither cost
calculation establishes a useful training-speedup claim at matched quality.

## Verification

- Re-ran the frozen audit read-only: six trajectories, 5,280 updates, 18
  discarded warmup updates, 36 curve points, source/optimizer mappings, RNG
  streams, labels, checkpoint manifests and cost arithmetic all pass.
- Independently recomputed all 36 prediction-set metrics and final control
  differences, cost ratios and acceptance flags. The final similarity deficits
  correspond to 40 / 13 net fewer correct predictions out of 5,000 than full.
- Reloaded and rescored six source-mode and six final-model conditions on
  CUDA. All saved logits reproduce exactly (maximum absolute error 0), and
  model states remain unchanged during inference.
- All 1,634 protected prior and raw files remain unchanged. No parameter
  updates or test-set scoring occurred during review. Current project pointers
  receive separately recorded before/after hashes.

Detailed evidence: `independent_audit.json`, `frozen_audit_recheck.json`,
`independent_curves.csv`, `independent_summary.csv`, `saved_state_rescore.csv`
and the preservation/seal manifests in this directory.

## Project and paper implication

Retain the result as a bounded negative comparison and a methodological lesson:
absolute recovery above the old baseline does not establish an advantage when
the full model receives the same additional training. The 21–23% marginal
recovery saving also must not be presented as a 21–23% total training saving.

Close this fixed adaptation recipe. Do not add epochs, choose an intermediate
checkpoint or relax the tolerance retrospectively. Return to distinct mechanism
design before allocating another training run. A new hypothesis would require
its own prospective protocol; none is prepared or executed in this review.
Stage14A's failed diagnostic and Stage13C's negative local-learning result remain
unchanged. This does not disprove token merging or local learning generally,
does not replicate/refute prior papers, and does not establish novelty. Two
development seeds and repeatedly used validation data limit generalization.
Publication remains on hold.
