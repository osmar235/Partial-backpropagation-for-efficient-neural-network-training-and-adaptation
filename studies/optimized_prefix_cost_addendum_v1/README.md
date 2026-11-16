# Optimized prefix/suffix cost — source addendum v1

Prepared, tested and frozen 2026-10-28; full scientific run pending.
Use source/README_OPTIMIZED_PREFIX_COST.md from the original project workspace.
Four boundaries, suffix-only and full refresh every 4/16 steps, against the
qualified BF16/automatic full-gradient execution policy. Four timing sessions,
two source checkpoints, 112 measured blocks / 10752 primary updates.

This measures training-block cost headroom, not quality or complete adaptation
speedup. Artificial CPU/GPU smoke results are implementation checks only.
Full and routed-full must reproduce exactly; all measured boundaries are retained.

The source closure includes prior reusable implementations, tests/protocols and
pinned upstream attention/license files. Provenance identifies byte-identical
copies. External installed-source hashes and original data/checkpoint requirements
are listed separately. Local absolute provenance remains; this is an organized
candidate for later release, not a standalone portable installation.

Checkpoints, CIFAR data, raw results and internal preparation/history scripts are
kept at their original project paths and excluded from this source capsule.
Internal work: protocol/optimized_prefix_cost/validate_preparation.py and
record_preparation.py; validation evidence: results/optimized_prefix_cost_preparation_v1.
No external publishing occurred; existing packages and evidence remain intact.
