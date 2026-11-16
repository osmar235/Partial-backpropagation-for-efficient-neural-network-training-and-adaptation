# Optimized attention/copy attribution — 2026-10-28

Bounded diagnostic authorized after closing the cut3/r4 adaptation recipe. Keep
full backpropagation and the qualified BF16/automatic policy. Identify remaining
operation costs before selecting an optimization; no new efficacy trial, novelty
claim, controller tuning, architecture campaign or automatic follow-up.

## Fixed conditions and control meaning

Original trained six-block ViT checkpoints1501/1502, batch sizes256/200, original
dropout and disabled-dropout diagnostics, two attention wrappers. The compatible
wrapper retains the qualified pinned upstream Attention plus the output
transpose/contiguous/transpose compatibility copy. The native_layout_control
returns that upstream Attention output directly. Same parameters/state mapping,
architecture, full gradients, BF16 forward, explicit FP32 loss and FP32 AdamW
moments, LR1e-4, wd.05, label smoothing.1. No compile/scaler/selective precision.

The control changes output strides. Downstream dropout can map the same random
stream to different logical elements; equal seeds do not mean matched masks.
Disabling all registered model dropout is a separate diagnostic, not deployment
or quality qualification. Keep model.train() in both modes. No equivalence is
assumed even without dropout: record actual losses/final states/RNG identities.
Removing our own compatibility overhead is not evidence of beating upstream
practice. This control is still our model with upstream attention, not a complete
independent timm architecture or a new algorithm.

For each seed/batch, select one fixed batch from the established training split,
augment once and apply MixUp once, with frozen seed formulas. Save sample indices,
MixUp lambda and input/target hashes. Repeat this same batch through measured
updates to isolate operation costs. No epoch, held-out scoring or accuracy claim.
Shared loader also loads validation arrays, immediately discarded; no test use.

For each case/session, construct a COMMON compatible anchor using four full
updates from source weights and fresh AdamW. Restore that same model/optimizer
anchor before both wrappers and every measurement mode. This is benchmark
conditioning, not resumption of archived17600-step moments or a quality endpoint.
Each arm receives eight discarded hardware-warmup updates before primary timing,
then exact restoration. All parameter gradients and moments must remain finite;
every optimizer counter must equal anchor age plus regime updates.

Four fresh-process sessions; eight seed/batch/dropout cases per session. Case
order is canonical, reversed, half-rotated, reversed half-rotated. Arm order
alternates by session and ORIGINAL canonical case index, giving two orders each
per case. Both primary arms run before any profiling within that case. Profiling
only in session0; potential carryover to later cases is a limitation to inspect.
Session preflight and final telemetry are retained; they do not establish absence
of interference throughout the run. No partial-session stitching on restart.

## Three accounting views, never mixed

1. Primary:64 plain updates per arm, synchronized only at block boundaries apart
   from the qualified loss.item() readback. Same existing update function, no
   phase events, profiler ranges or module hooks. Wall clock includes RNG/zero,
   forward/FP32 loss/readback, full backward and AdamW. Fixed-input preparation,
   data/model loading, anchor construction, warmup/restoration, checks, artifact
   writes and profiler exports are excluded. This is operation-block cost, NOT
   complete adaptation. The wrappers include minimal diagnostic branch plumbing.
2. Session0:64 matched CUDA-event/host-phase updates from the same anchor. Phases:
   seed_zero, forward_loss, loss_readback, backward, optimizer. Events bracket
   stream intervals, which include launch gaps, not just active kernels. CPU
   phase times are asynchronous except readback. Do not add CPU and GPU times.
   Require identical losses/final model/optimizer/RNG to the64-step plain block.
3. Session0:four short plain updates and four matched profiler updates from the
   same anchor, profiler warmup1/active3. Require exact within-arm learning/state/
   RNG agreement. Profile CPU and available CUDA activity, operator shapes, forward
   module ranges and a specific compatibility-copy range. Backward attribution
   is by phase/operator, not assumed from forward module labels. Export traces.

Profile overhead includes initialization, warmup, active recording and teardown;
report it against short plain. Event overhead is relative to64-step plain. Do
not rank wrappers by profiler or event wall time. Availability of CUDA attribution
requires recorded kernel events and positive linked device time. Missing CUDA
activity means unavailable, not zero cost; retain CPU/event diagnostics.

Group CPU-event self times and linked device self times; exclude raw CUDA events
from those sums to avoid double counting. Broad copy-like op totals are descriptive
and include other copies/casts/optimizer work. The explicit copy range identifies
FORWARD compatibility work only; no claim that it identifies all backward costs.
Profiler shares are not removable end-to-end wall-time fractions. Record actual
SDPA dispatch in active profiling, without asserting flash from policy name.

Full budget:64 primary blocks/4096 updates;16 event blocks/1024 updates;16 short
plain blocks/64 updates;16 profiler blocks/64 updates (16 warmup+48 active);32
common anchors/128 updates;64 arm warmups/512 updates. Total5888 diagnostic full
updates.16 Chrome traces. No new learned-method quality result follows.

## Prospective interpretation and audit

Report every paired primary saving, median/range, positive counts and order.
Also report cross-wrapper learning identity and first loss difference. Four
repeats support descriptive consistency, not significance or a winning layout.
No newly chosen5%/10% threshold. Qualifying an optimization requires a distinct
implementation rationale, correctness and complete-workflow comparison against
the strongest established implementation after this profile is reviewed.

If compatibility work accounts for the observed difference, label it adapter
overhead; it is not a novel speedup over upstream. If other costly copies or host
overhead are material, propose at most one change. If little actionable work
remains, close this optimization branch and reassess the paper contribution.
Do not select another training campaign automatically. Earlier failures stay closed.

Save configuration/input/anchor hashes, losses, final state hashes, ages, event
phases, operator tables, raw traces and telemetry. Final checkpoints are not
archived for each diagnostic regime; finite-state/age assertions run live. The
saved-data auditor verifies manifests, schedules, counts, instrumentation identity
and recomputes timings/operator aggregates, without rerunning training. Unexpected
errors preserve failed attempts. Restart reruns the whole interrupted session;
report any retries, and do not treat retry-free timing consistency as established
when sessions were retried. Completed rerun audits saved files only.

Preparation uses tiny artificial CPU/FP32 and CUDA/BF16 workloads. It may inspect
real checkpoint structures read-only, but performs no real forward/backward/update.
Use existing installed torch2.11 environment; no dependency updates required.
