# Optimizer sensitivity results — paper-support addendum v1

Reviewed 2026-10-27. Read REPORT.md for findings and limits. Both frozen and
independent raw-tensor audits pass. No model gradient or optimizer step was
rerun during review. Fresh AdamW amplification is supported; recent history
reduces disagreement; archived history raises relative/worst-coordinate errors
while lowering absolute L2 discrepancy on the training batch. See norm_scale_check.json. Post hoc attention
key-bias localization motivates a functional-impact test, not a quality claim.

All 132 comparison rows and aggregate/coordinate/dispatch/source summaries are
included. Run `python -B recompute_summary.py` here to reconstruct the 11 grouped
summaries with the standard library. It does not redo raw tensor or provenance
checks. original_workspace_review.py is a provenance copy requiring the original
workspace, project Python/torch and sibling record_preparation helper. Run its
original protocol/optimizer_sensitivity location only to create a new review;
it refuses to overwrite existing output. Several GB of raw tensors remain at
their original results paths and are not copied into this package.

Frozen diagnostic code: ../optimizer_sensitivity_addendum_v1/. Earlier packages
and studies remain unchanged. No next experiment is prepared, and nothing is
published. SHA256SUMS.json records the integrity of this results addendum.
