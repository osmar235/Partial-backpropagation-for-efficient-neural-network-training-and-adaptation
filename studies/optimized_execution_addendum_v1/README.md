# Optimized execution — paper-support source addendum v1

Prepared and frozen 2026-10-27. Full experiment pending. This capsule contains
the runner, auditor, tests, prospective specification, source closure, exact
pinned timm source/license, freeze, and local compiler capability evidence.
Run from the original workspace using README_OPTIMIZED_EXECUTION.md. Provenance
retains original absolute paths; this is not a portable training release.
Installed torch/torchvision dependency files are identified by their frozen
hashes in EXTERNAL_DEPENDENCY_HASHES.json, rather than redistributed here.

Six eager arms compare native/reference attention with FP32/math, BF16/math,
and BF16/automatic dispatch. Reference FP32/math is required. Compilation is
unavailable in the local tiny probe (TritonMissing); no compiled timing or
environment installation is substituted. No checkpoint or dataset is included.

Three CPU checks passed with one CUDA-only skip; four CUDA checks passed.
Both tiny workflows audited; only reference FP32/math qualified. Native FP32
failed the tiny post-Adam bias tolerance; all tiny CUDA BF16 arms failed the
fixed parameter sanity bound. Automatic BF16 dispatched to memory-efficient
SDPA. Separate BF16 trial-function tests passed without waiving those gates.
Tiny timings are implementation diagnostics, not scientific performance evidence.
Full-size eligibility remains unknown. No tolerance was relaxed after seeing
results. BF16 sanity qualification cannot establish real-data quality.

Earlier public-support packages remain unchanged. Preparation helpers, logs,
tiny raw data and compiler caches stay in the project research record outside
this source capsule. SOURCE_PROVENANCE.json maps originals and SHA256SUMS.json
seals this folder. Nothing has been published. No full study ran in preparation.
