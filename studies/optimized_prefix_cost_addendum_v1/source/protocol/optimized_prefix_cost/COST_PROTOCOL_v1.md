# Optimized prefix/suffix cost feasibility — 2026-10-28

Prospective bounded cost screen following the qualified BF16/automatic adaptation
baseline. No quality campaign, adaptive gradient method or publication claim.
No changes to earlier gates, tolerance budgets or completed evidence.

## Fixed workload

Two archived source checkpoints 1501/1502, same six-block RGB32 ViT and pinned
reference attention. BF16 forward, automatic SDPA, explicit FP32 cross-entropy,
FP32 parameters/moments, AdamW, learning rate 1e-4, weight decay .05, MixUp .2,
label smoothing .1, batch 256, original train-mode dropout. No compile, scaler,
clipping, coordinate exemption, input/activation caching or new dependency.

Use only the established 45000-example training partition. Loading the existing
shared data helper also loads validation arrays, which are discarded; no validation
inference, test access, accuracy selection or quality qualification in this study.

In each fresh session, construct one common anchor per source by 16 full updates
from source weights and fresh AdamW, using a separate fixed calibration stream.
All arms restore that identical model and optimizer anchor, including counters.
Calibration is benchmark conditioning, not an extension of the source optimizer
history and not a trained-quality result. Store/compare anchor hashes across arms.

Fourteen fixed arms: full native forward/update control; routed-full control;
four boundaries after encoder blocks 1, 3, 5, 6, each with suffix-only updates,
full refresh every 4 updates, and full refresh every 16 updates. Cut k freezes
patch projection, class/position embeddings and blocks 1 through k. Final encoder
layer norm and classifier always train, including cut 6. No altered forward code.
All arms recompute the entire forward for each current augmented input. Frozen
prefix dropout remains in train mode. No gradients are propagated through the
frozen prefix; inactive parameters have grad=None, no decay or optimizer aging.
A full refresh occurs at zero-based indices 0, R, 2R, ...; suffix-only has none.

Each arm: 16 discarded warmup updates (first profiled), exact anchor restoration,
96 measured updates, then restoration and a separate 16-update instrumented pass.
The latter replays the first measured updates with individually synchronized
augmentation, routing/RNG/zero-grad, forward/loss, scalar readback, backward and
optimizer components. Timing synchronization perturbs execution; these component
times cannot be substituted into or subtracted from the primary clock.

Four fresh-process sessions: ordered arms, reversed, half-rotated, reversed
half-rotated; sources reverse on alternate sessions. Each arm has the same mean
ordinal position and mirrored placements. This is not full permutation coverage
or a hardware-drift correction. Fixed random streams measure timing repeatability,
not additional independent model seeds. 112 measured blocks / 10752 primary
updates, plus 128 anchor, 1792 discarded warmup and 1792 component updates.

## Accounting and interpretation fixed before measurement

Primary metric: synchronized wall time of the 96-update training block, including
batch indexing, augmentation, MixUp, RNG, routing, zero-grad, forward, explicit
FP32 loss/readback, backward, AdamW and in-memory trace collection. CUDA synchronization
at block boundaries; scalar loss readback each step follows the qualified baseline.
Imports, source/data loading, anchor construction, model setup, profiling/warmup,
restoration, component diagnostics, telemetry, checks and artifact writes excluded.
This is not complete adaptation timing. All fixed arm preparation overhead is
excluded symmetrically. Report the actual denominator prominently.

Compare each arm with same-source/session full baseline; report all paired values,
median/range and positive-session counts. Four timing repeats support descriptive
consistency only; no significance, winning-boundary, new 5%/10% cutoff or automatic
candidate selection. A useful next mechanism requires review of costs AND a new
prospective quality experiment. Negative findings are retained.

Suffix-only savings estimate optimistic compute headroom assuming no need for
prefix learning and no extra decision/approximation mechanism. For each R, report
the zero-extra-overhead mixture T_full_routed/R + (1-1/R)*T_suffix. Compare measured
periodic updates to this projection; retain signed residuals (noise and different
trajectories prevent attributing all residual to routing). This is not a rigorous
universal bound, a gate cost, or evidence that the refresh schedule preserves quality.
Actual periodic blocks charge flag changes, real full refreshes and optimizer work.

Record allocated memory (including shared resident data/anchors), backend dispatch
in the first warmup only, streams, losses, state hashes and integrity checks. Full
versus routed-full must exactly match per-step learning, final weights and moments;
otherwise block interpretation. Check prefix gradient absence, finite active
gradients/states, expected per-parameter optimizer ages and unchanged prefix weights
for suffix-only. These are runtime assertions; final weights are not archived for
every arm. The auditor independently reconstructs timings/schedules, verifies
manifests and recorded control identities; it does not recreate training states.

Abort on unexpected/nonfinite errors and preserve attempts. Restart skips sealed
sessions; an interrupted session reruns all arms in a fresh attempt. Never splice
arms across processes. Any retried session blocks the descriptive consistent-headroom
flag and requires review; all attempts remain available. Completed rerun audits
without training. CPU tiny smoke checks FP32 plumbing; CUDA tiny smoke checks BF16
automatic. Artificial smoke cannot establish scientific cost/quality results.
