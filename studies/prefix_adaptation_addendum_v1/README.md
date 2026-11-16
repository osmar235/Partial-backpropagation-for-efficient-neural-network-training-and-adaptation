# Prefix adaptation quality/cost — source addendum v1

Prepared and frozen 2026-10-28. Full scientific run pending. Use
source/README_PREFIX_ADAPTATION.md in the original project workspace.

Three arms under qualified BF16/automatic execution: full, frozen prefix through
block3, and full prefix refresh every4 steps. Five epochs, two trained sources,
six counterbalanced timing sessions. Both matching scoring modes must meet the
0.30-point quality margin; primary time is complete adaptation. Starting weights
are original sources with fresh AdamW, not cost-screen endpoints/calibration.

The frozen control tests whether refresh is needed for this fixed screen.
Conventional freezing/scheduling is not a novelty claim. Earlier Stage9E and
other failed gates remain unchanged; no automatic cut/interval rescue sweep.

Seven CPU and seven CUDA tests pass; both tiny end-to-end workflows pass and
rescore exactly. Real source inference reproduced saved predictions exactly,
with no real backward or optimizer updates. See project preparation report.

SOURCE_PROVENANCE lists byte-identical copies including the reusable dependency
closure and pinned upstream attention/license. DATA_REQUIREMENTS and external
hashes identify original artifacts and installed sources. This organized source
capsule retains local absolute provenance; a portable installation requires
separate release work. No checkpoints, CIFAR data or internal preparation history
are bundled, and nothing has been published externally.

Internal derivation/validation/preservation helpers remain in
protocol/prefix_adaptation; logs and smoke results remain in project results.
