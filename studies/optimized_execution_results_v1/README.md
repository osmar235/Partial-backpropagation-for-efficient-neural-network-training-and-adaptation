# Optimized-execution results — paper-support addendum v1

Reviewed 2026-10-27. REPORT.md contains findings, failures and scope. Frozen and
independent audits pass. Both FP32 arms qualified; all 64 BF16 evaluations failed
only the declared parameter-difference gate and were excluded from timing.
The established FP32 reference saved 5.78–9.48% in this operation screen.
This is not a BF16 accuracy failure, BF16 speed result or new-method claim.

All 192 trial records and numerical/order/dispatch summaries are included.
Run `python -B recompute_timings.py` here to reproduce four timing summaries
using the standard library. This does not reconstruct raw numerical tensors
or perform the provenance/eligibility audit. original_workspace_review.py is
a provenance copy; run its original protocol/optimized_execution location only
when reconstructing a new review in the original workspace (it refuses to
overwrite existing review output and imports record_preparation from there).

Code/protocol capsule: ../optimized_execution_addendum_v1/. Raw outputs and
internal preparation records remain at their original paths. Earlier public
packages are unchanged. SHA256SUMS.json seals this results addendum. No release
occurred. No experiment was rerun for review. A small optimizer-sensitivity
diagnostic is recommended next; no follow-up experiment is yet prepared.
