# Matched-state runtime attribution — 2026-10-25

## Purpose and scope

The fixed sampled weight-gradient implementation failed its native-time gate in
all four conditions. It remains parked. This artificial diagnostic asks where
time is spent before selecting another design. It is not an accuracy experiment,
a new candidate screen, or evidence of novelty. No real data/checkpoint is read.

Compare native PyTorch, exact all-token custom autograd, and the same parked
16-of-64-patch custom gradient. Preserve the full forward and dense input
gradient. Keep the previous FP32, deterministic math-attention execution settings;
this does not establish the best attainable performance of native PyTorch.

## Fixed workload

The full diagnostic uses seeds 99051/99052, batches 256/200, and all three arms:
12 conditions. Each uses the six-block, width-192 ViT with 65 tokens, random
weights and fixed artificial inputs. Perform three burn-in AdamW updates; retain
the resulting model and optimizer state as the common anchor for that condition.

Restore that anchor before each regime. Three rounds compare uninstrumented
plain updates with phase-event updates, counterbalancing their order. Each
regime has four updates. One profiler regime then runs four matched updates:
one profiler warmup plus three active updates. Total: 336 measured artificial
updates plus 36 burn-in updates. No efficacy or speed pass threshold is applied.
Across all seven regimes within a condition, losses, final parameter and AdamW
hashes, and CPU/CUDA random-number states must match exactly.

CPU/CUDA smoke checks use one seed, batch three, a reduced two-block model, one
burn-in update, one plain/event round and one profiler regime (one warmup plus
one active update). Each smoke has 18 measured updates and three burn-in updates.
Smoke timings must not be used to infer full-workload costs.

## Three distinct measurements

1. Plain uses the unchanged previous update function. Its synchronized wall
   time is the timing reference for instrumentation overhead.
2. Events bracket seed/zeroing, forward, loss/readback, backward and optimizer.
   Host timings are asynchronous except existing synchronization such as
   loss.item(). CUDA events measure stream intervals, including launch gaps;
   no per-phase synchronization is added. Their overhead is reported against
   the matched plain regime. These are not pure kernel-compute durations.
3. The PyTorch profiler records operator names, shapes, CPU self times and
   available linked GPU self times. Forward-only module ranges distinguish
   attention, MLP, normalizations, patch embedding and classifier. Backward is
   attributed by phase and operator/shape; it is not directly decomposed into
   per-module weight/input gradients. Do not infer such attribution from a
   forward label. Profiling is run after the plain/event measurements.

Profiler-session wall time includes initialization, warmup, collection and
teardown. Do not rank implementations by it. Trace export, hashing and CSV
writing are outside timing. CPU/GPU totals overlap; nested module ranges and
operator self times are not additive. No stack or memory profiling is enabled.
Missing CUDA operator activity is marked unavailable, never zero GPU cost;
CUDA-event phase timings and CPU attribution still remain useful. If collection
fails entirely, retain the partial attempt and review the error before proceeding.

Official references: [PyTorch profiler](https://docs.pytorch.org/docs/2.11/profiler.html),
[CUDA events](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html).
Shape tracing itself can perturb execution; matched timing controls are essential.

## Integrity, execution and interpretation

Freeze source, specification, execution settings and environment fingerprints
before the full run. Stop other GPU workloads. Retain telemetry before/after.
Finished conditions and the root output have hashed completion markers. A rerun
validates and skips completed work; incomplete attempts are preserved separately.
The independent auditor reconstructs inputs, regime order, pairings, counts,
hash checks and summary tables. It does not rerun the scientific workload.

Read phase tables together with plain/event overhead, operator shapes and traces.
Identify an expensive component and a plausible way to remove its work before
proposing a new algorithm. This diagnostic alone cannot show real-data quality,
complete adaptation savings, or universal bottlenecks. There is no automatic
follow-up experiment, fraction search or real-data launch. Publication stays on
hold and all previously frozen evidence remains intact.
