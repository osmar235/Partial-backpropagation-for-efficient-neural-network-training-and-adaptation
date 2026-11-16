# Competitive attention results — paper-support addendum v1

Completed screen reviewed 2026-10-26. Read REPORT.md for interpretation and
limitations. The pinned upstream reference is generally faster than our unbind
adapter in this artificial FP32/math-attention screen. All numerical gates pass;
the upstream reference is tolerance-matched, not bitwise identical. This is not
a real-data quality result or a new-method superiority claim.

The code/protocol capsule remains in ../attention_reference_addendum_v1/.
Earlier packages remain unchanged. This results addendum contains all 288
exported trials, independent comparisons, numerical/mechanism/order/telemetry
summaries and the review. SHA256SUMS.json records this folder's integrity.

From this folder, `python -B recompute_timings.py` recomputes all twelve timing
summaries using only the Python standard library. It does not redo the source,
parameter or trajectory audit. original_workspace_review.py is a byte-identical
copy of the independent reviewer script for provenance: execute its original
protocol/attention_reference/review_completed_v1.py location only when reconstructing
a new audit in the original workspace; it deliberately refuses to overwrite an
existing review folder. It depends on the original frozen evidence paths.

No training or timing experiment was rerun for review. No public release has
occurred. No next experiment is prepared; optimized-execution feasibility with
this upstream reference as a required control is the recommended next design.
