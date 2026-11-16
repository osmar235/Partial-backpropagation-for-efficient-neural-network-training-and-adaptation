# Stage13A decision — 2026-10-23

Integrity PASS; no efficacy pilot justified from this diagnostic.

The full-reference mean score-loss improvement is negative in both source
states. Every one of96measured updates worsens its score-batch loss. Under the
frozen protocol, the usefulness comparison is therefore inconclusive, not a
valid rejection of local learning relative to a useful full-gradient reference.

The resource measurements remain descriptive: local mean update-time savings
2.41% /9.36%, incremental allocated-memory savings6.38% /6.26%. Both states miss
the required10% time OR20% incremental-memory reduction. Final-block gradients
agree, while early local gradient cosine is weak. Cosine is not an efficacy gate.

All first-trial saved before/after score losses reproduce on reload. Independent
arithmetic and frozen source/state audits pass. No training or held-out scoring
was performed during review. Details: results/stage13A_review_v1/REPORT.md.

Park this fixed local-head/block design; no long pilot or automatic retuning.
Do not change LR/source/calibration or call a smaller negative loss change an
improvement. A future revisit requires a separately frozen, reference-qualified
diagnostic. This does not rule out local learning in general.

Next queued idea: later feature/token compression, requiring its own protocol.
No new experiment command in this review. Earlier positive tradeoffs and their
limitations are preserved. Publication remains on hold.
