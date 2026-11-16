# Competitive attention reference, first screen — 2026-10-26

Prospective operation-level screen, artificial inputs only. No training-data
accuracy or end-to-end adaptation claim. This is the first part of the accepted
competitive-baseline study; mixed precision, compilation and alternative backend
policies need separate prospective protocols after this controlled comparison.

Three arms: original native MultiheadAttention (`math`); unchanged measured
`qkv_unbind`; and the exact Attention class from timm v0.9.2, resolved commit
`3d05c0e86f2f4c57bf495468aa3f8a7d3487c986`. Full upstream file and Apache-2.0
license are archived in third_party/timm_attention_v0_9_2 with SHA-256 provenance.
The upstream class AST is executed unchanged; its fused-attention switch is
fixed True to call SDPA. qk_norm=False, attention/projection dropout=0, biased
projections, and identical native parameter objects and state names. The rest
of the model is unchanged. This is an upstream attention-layer comparison in
our model, not a benchmark of the whole timm package or its latest release.
The adapter restores the native output strides by transposing, making the
sequence-first output contiguous, and transposing back. This boundary copy
is included in timed work. It preserves the downstream dropout mask mapping;
equal RNG states alone do not guarantee equal masks for differently strided
tensors. The upstream forward itself remains unchanged.

The reference projects batch-first and reshapes/permutes/unbinds directly;
our adapter retains native sequence-first GEMM and packed contiguous copy.
Thus this compares implementation families; the difference cannot be attributed
solely to unbind. No claim of a novel unbind primitive is permissible.

All arms: same two initialized models, artificial images/labels, stochastic
streams, AdamW updates, full gradients, FP32, TF32 off, strict deterministic
execution and forced math SDPA. Shape: original six-block width-192 model,
65 tokens, batch 256/200. Four fresh processes, six round blocks per condition.
Within each session/condition, the six permutations of three arms are used
once each, rotated by condition/session. Preserve failed-arm slots; the order
of remaining arms follows the planned full schedule. This balances positions
when all arms qualify. Log actual order if numerical gates exclude an arm.

Before timing each condition: three updates check logits, loss, every parameter
gradient, post-Adam parameters and both moments against native, under the exact
existing QKV tolerances. No key-bias exception or post-result tolerance changes.
Repeat each arm to verify exact state/RNG determinism; record actual math SDPA
and split/copy operations through a separate profiler probe. All parameters
must receive finite gradients and keep native parameter/state mapping.
Numerical failure is a recorded outcome, not a fatal audit error. Failed arms
are not eligible for timing or a quality-preserving efficiency claim.

Each trial: construct identical state; five discarded warmup updates; restore
initial model and fresh AdamW; synchronize and separately measure first update
and 32 steady updates. Loss extraction/synchronization behavior is the same
in all arms. Report steady milliseconds/update (primary), first update, setup,
warmup/restore and allocated memory (secondary). Timed work excludes data
generation, profiling, correctness snapshots, hashes and evidence serialization.
This is not a full-workflow clock. If all arms qualify: 288 trials, 9,504 measured
updates and 1,440 discarded warmup updates, plus separately identified checks.

Report native versus unbind, native versus reference, and reference versus
unbind paired reductions for each seed/batch/session (median of six paired
ratios), then descriptive across-session medians/ranges and positive counts.
Four shared-machine sessions are not independent model replications. No
confirmatory p-value, new minimum saving threshold or automatic real training.
The original 10% gate remains failed. Include every pair, without outlier removal.

Source/spec/environment freeze precedes the full run. Each session runs in its
own child process. Completed conditions are hash-checked and skipped on rerun;
incomplete attempts remain preserved under a new numbered attempt. Preflight
and before/after telemetry are recorded. The audit verifies counts, scheduled
orders, source/input/state matches, numerical eligibility and completion hashes.
Interrupted/restarted sessions must be labeled, not treated as fresh unfragmented
sessions. A completed screen requests review; it does not select a follow-up.
