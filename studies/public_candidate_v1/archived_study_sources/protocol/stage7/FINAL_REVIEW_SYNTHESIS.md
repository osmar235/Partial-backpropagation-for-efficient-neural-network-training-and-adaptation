# Completed Stage 7 evidence and manuscript revision map

Updated 2026-10-18. All prospectively listed phases are complete: 7A archived
audit, 7B mechanism (36 runs), 7D CIFAR-100 (12), 7C sensitivity (36), and optional
7E AdamW (12). Total: **96 new substantive training runs**. The original Stage
6B nine-seed confirmation and Stage 5 failure boundary remain separate and intact.
This record is the manuscript revision plan, not a claim that manuscript files
have already been revised or submitted.

## Evidence to carry into the paper

| Study | Main finding | Claim boundary |
|---|---|---|
| Stage 6B, original confirmation | 10.24% training-wall reduction versus Adam; original accuracy non-inferiority criterion passed | Original workload and preregistered analysis only |
| 7A, archived audit | Original mean partial fraction 35.990%; supplies archived epoch profile | Existing timing does not resolve full/partial backward |
| 7B, six-seed mechanisms | 8.52% less wall time than Adam, accuracy -0.108 pp (-0.580, +0.363) | No clear loss-gate advantage over random/periodic controls; replay competitive |
| 7D, CIFAR-100 | 9.98% less wall time, accuracy -0.695 pp (-1.358, -0.032) | Dataset transfer with accuracy cost; no new architecture or preserved-accuracy guarantee |
| 7C, sensitivity | Removing refresh/larger prefix buys time at an accuracy cost; no clear EMA improvement in both outcomes | Exploratory; substantial timing variation; no post-result controller replacement |
| 7E, AdamW | 9.66% less wall time, accuracy +0.028 pp (-0.106, +0.163) | Fixed weight-decay recipe robustness; no equivalence or new formal non-inferiority declaration |

Stage 7 intervals are exploratory paired t intervals and unadjusted for multiple
comparisons. Keep phase-specific pairs and references explicit. Do not pool
absolute training durations or compare unmatched seeds to manufacture Adam
versus AdamW, CIFAR-10 versus CIFAR-100, or hyperparameter superiority claims.

## Manuscript revision map

1. **Abstract:** retain the original confirmation, add AdamW robustness and the
   CIFAR-100 accuracy cost, and avoid claiming the gate uniquely creates the
   benefit. Select detail to fit the journal's length requirement without hiding
   negative/inconclusive mechanisms.
2. **Methods:** distinguish original confirmation from prospective post-confirmation
   studies. Explain archived epoch-matched blind masks, no adjacent partials,
   freeze/replay state handling, variant-specific rho/partitions, AdamW decay,
   algorithm-specific warm-up, cyclic orders, exact reference/sign conventions,
   timing regions, and observed epoch-crossing time-to-quality.
3. **Results:** retain original Stage 6 estimates unchanged. Add complete mechanism,
   sensitivity, CIFAR-100 and AdamW tables with paired intervals. Report realized
   q, skip streaks, full/partial backward timers, and all predeclared target
   crossings, including unreached targets. Put per-seed/raw detail in supplement.
4. **Mechanism discussion:** matched random/periodic controls were competitive;
   no unique loss-signal benefit was shown. Moment continuation improved
   accuracy over freezing (+0.577 pp, exploratory interval +0.219 to +0.935),
   but replay remained competitive. Gate decisions differ when trajectories
   change; do not describe these as a common-mask causal decomposition.
5. **Sensitivity:** no forced-full gave -0.517 pp with 5.62% less time; larger
   prefix gave -0.823 pp with 4.98% less time. Stem-only was 9.62% slower for a
   small uncertain +0.105 pp. Long skip streaks reached 93. Preserve the original
   operating point; do not select a new best variant from these data.
6. **Timing limitations:** disclose seed 57's faster regime and within-run
   drift in seed 58 EMA 0.90. Paired cyclic order is useful but cannot remove
   changing conditions between sequential runs. No telemetry establishes the
   cause. Retain all seeds; keep small timing claims cautious.
7. **Generality:** CIFAR-100 is a second dataset in the same architecture family.
   AdamW is an extra recipe with decoupled weight decay, not a tuned modern
   benchmark or demonstrated Adan-based LGPB implementation. No energy savings,
   other GPU, BatchNorm, mixed precision or multi-GPU benefit was established.
8. **Independence and literature:** distinguish new run seeds from an untouched
   evaluation dataset. Adan is a comparator, not a partial-Adan controller.
   Preserve the related-work checks recorded in protocol v1.1, including example
   selection versus prefix-gradient omission, without cross-paper speed rankings.
9. **Submission metadata:** actual journal and submission status remain an
   author-verification item. The AI review is not established as formal journal
   peer review. Do not invent a submission event or resolve the conflicting
   headers by assumption.
10. **Reproducibility:** preserve original release artifacts. Prepare new Stage 7
    supplementary materials as a separate dated version when preparing the
    revised manuscript; no old DOI/archive overwrite or public release is
    implied by this local update. Seeds 51–74 used in these phases are now seen
    (seed 50 remains diagnostic/development).

## Records

Per-phase audited reports: `results/stage7B_review_v1/REPORT.md`,
`results/stage7C_review_v1/REPORT.md`, `results/stage7D_review_v1/REPORT.md`,
`results/stage7E_review_v1/REPORT.md`. Their audit JSONs record input provenance
and trace checks. Training outputs remain in their original Stage 7 directories.
No further training is required by the completed plan.
