# BF16 matched adaptation — source addendum v1

Prepared and frozen 2026-10-27; full study pending. This capsule contains the
runner, checkpoint/stream auditor, six tests, CMD launcher, prospective protocol,
specification/freeze and required local source closure, including pinned upstream
attention source and license. Earlier evidence and packages are unchanged.

Use source/README_BF16_CONTINUATION.md from the original workspace. The freeze
retains absolute provenance, so portable public packaging is still pending.
External installed dependencies and local checkpoint/data requirements are
identified separately; checkpoints, CIFAR data and raw outputs are not copied.
Internal derivation/preparation helpers and draft smoke files remain outside
this capsule. Do not publish the whole project root.

Six CPU and six CUDA tests passed. Tiny CPU/CUDA workflows audited 4/12 trained
artificial trajectories, with 48/144 updates and exact final common/own-policy
rescoring. Completed restart skips training and runs inference audit only. Both
real source checkpoints reproduce all archived validation logits exactly with no
gradient or optimizer update. These are readiness checks, not scientific results.

Prospective scope: 36 five-epoch trajectories from two existing checkpoints,
three execution policies and six counterbalanced sessions. All parameters train,
AdamW is reset equally, and loss is explicitly FP32 outside autocast. Quality
must stay within 0.30 percentage points in both evaluation modes, every session
and both checkpoints; reference readiness is compulsory. Complete-cost savings
must be positive in every session/source, with the frozen adjusted direction
test. Repeated timing sessions are not additional quality seeds. No automatic
follow-up, gate waiver, new-method claim or publication readiness is implied.
SOURCE_PROVENANCE.json and SHA256SUMS.json identify the capsule contents.
