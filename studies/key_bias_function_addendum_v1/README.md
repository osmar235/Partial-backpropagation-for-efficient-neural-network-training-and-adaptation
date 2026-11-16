# Key-bias functional diagnostic — source addendum v1

Prepared and frozen 2026-10-27. Full scientific run pending. This local capsule
contains the runner, stored-logit auditor, four tests, CMD launcher, prospective
specification/protocol/freeze, required source closure and pinned upstream
attention source/license. No existing paper-support package was changed.

Use source/README_KEY_BIAS_FUNCTION.md in the original workspace. Absolute
provenance remains in the freeze; this is not a portable release. Installed
dependency hashes and local raw-data/checkpoint requirements are listed in
EXTERNAL_DEPENDENCY_HASHES.json and DATA_REQUIREMENTS.json. Data, checkpoints,
saved gradients and smoke tensors are not copied into this capsule.

Four CPU and four CUDA tests passed. Complete tiny workflows audited 8 CPU and
120 CUDA component comparisons, including exact replay and component isolation.
CUDA recorded 420 finite forward outputs. These are artificial checks, not
scientific results. Real source/input consistency was checked without full-model
evaluation or optimizer steps. The full diagnostic is prepared for the author.

Compare full, key-only and key-restored updates across two frozen trained
sources, two anchors/histories, two training-only batches and three evaluation
policies. No new gradients. Uniform FP64 scoring is checked against explicit
FP32 scoring; original execution loss is reported separately because artificial
validation found an autocast loss-arithmetic difference in this runtime.
Earlier BF16 gate failures and scoped QKV speed results remain unchanged.
This diagnostic cannot establish long-run quality, speed, novelty or readiness
to publish. Internal preparation helpers and draft failures remain outside this
capsule. SOURCE_PROVENANCE.json and SHA256SUMS.json identify its contents.
