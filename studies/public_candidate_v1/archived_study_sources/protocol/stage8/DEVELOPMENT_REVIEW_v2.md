# Stage8 v2 development review and evaluation decision

2026-10-19. Review performed before the real v2 freeze and before reserved
evaluation. Detailed audit: `results/stage8A_review_v2/REPORT.md` and `audit.json`.

| Development seed | Full AdamW final validation | LGPB final validation |
|---:|---:|---:|
|1001|85.46%|70.48%|
|1002|86.08%|72.40%|
|Mean|85.77%|71.44%|

Baseline quality passes the unchanged mean85%/minimum83% guard. LGPB has a
large adverse development gap, **-14.33 percentage points**. This is retained
evidence, not a reason to discard these seeds or declare transfer successful.
The curves separate early and remain separated. Best epochs do not replace finals.

All four runs pass integrity checks:1200epochs/211,200updates, source/config
hashes, exact split, gate/EMA, nonadjacency, LR, MixUp lambdas, timing, targets,
checkpoint SHA-256 and52,800-step optimizer counters including every prefix
parameter. All saved weights/state are finite. Paired initial states match.
Independent CPU checkpoint rescoring exactly matches the saved accuracies;
losses agree within1e-4. Exact fresh suffix gradients were verified for mixed
targets at each final checkpoint. No training or test-data loading was required.
These checks found no implementation defect, but cannot exclude every defect.

## Descriptive clue, not causal attribution

Mixing strength is min(lambda,1-lambda). In seed1001 its mean is0.01645 on
LGPB partial steps and0.15601 on full steps; seed1002 gives0.01646 and0.15538.
Thus the loss gate selects partials mostly on weakly mixed minibatches. This
association can be computed from the logged losses/lambdas alone. It does not
show that MixUp caused the accuracy gap, nor isolate gating from omission,
prefix choice or optimizer-state handling. There is no no-MixUp LGPB arm here.

The two realized partial fractions are39.73%/39.78%. Development ordering was
baseline then LGPB, so timing differences are descriptive with possible drift.
Six reserved seeds with balanced four-arm ordering provide the planned
comparison. Independent development-profile matching need not exactly match
each evaluation trajectory's realized partial rate; report that difference.

## Decision

Continue the already specified24-run evaluation **without a third recipe or
controller change**, preserving this warning. Baseline readiness was the
declared quality gate; a favorable LGPB calibration outcome was not a required
gate. Adding one retrospectively would select the study for favorable evidence.
The four-arm comparison can characterize a limitation of this method, prefix
and recipe. It does not guarantee a positive transfer result and cannot support
a categorical statement about all transformers or all augmentation schemes.

Settings/spec/source files remain unchanged. A real prospective freeze was
created only after the audit:
`protocol/stage8/FROZEN_EVALUATION_v2.json`, identifier
`fea1b99e9f6abf58a5f14c8cf1624c7e6cf2bd6c407417f045e3b0c36e02b5e6`.
It binds source/spec hashes, all completed development inputs, LR0.001 and the
300-epoch periodic profile. This local freeze is not a public preregistration.
No reserved evaluation trajectory was started during review.

Report both failed v1 baseline screening and adverse v2 LGPB calibration in
the eventual release/supplement. Keep new Stage8 inference separate from
Stage6 confirmation and all completed Stage7 evidence. No revised numerical
claim has been inserted into the working manuscript before evaluation.
