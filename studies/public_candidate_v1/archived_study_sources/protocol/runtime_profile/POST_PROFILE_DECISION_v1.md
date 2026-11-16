# Runtime profile review and next design — 2026-10-25

## Decision

Keep the fixed sampled-weight implementation parked. The completed runtime
diagnostic supplies a systems explanation, not a positive method result. The
next recommended step is an **exact-gradient execution feasibility study**:
check attention backends and tensor layout before another approximation. This
document records a proposal; it does not freeze or launch the next experiment.

## What the measurements support

All 12 conditions completed; 336 measured updates and 36 burn-in updates. The
frozen read-only auditor and independent reconstruction passed. Instrumented
losses, model/optimizer states and RNG states match their plain controls exactly.
No real-data accuracy or complete adaptation outcome was measured.

| Seed | Batch | Native median ms | Sampled median ms | Sampled slowdown |
|---|---:|---:|---:|---:|
| 99051 | 256 | 29.050 | 30.504 | 5.01% |
| 99051 | 200 | 22.579 | 25.963 | 14.99% |
| 99052 | 256 | 29.113 | 30.247 | 3.90% |
| 99052 | 200 | 22.929 | 25.045 | 9.23% |

These are ratios of uninstrumented median timings, not the previous cost gate's
median paired ratios. The diagnostic has three short repeated timing rounds per
arm and condition; these are not independent training replications. Its anchors
match within an arm, not across approximated and exact trajectories after burn-in.
It corroborates the earlier failure without replacing that prospective gate.

Native CUDA-event medians: forward 7.30–9.66 ms, backward 14.10–18.23 ms,
optimizer 0.72–0.93 ms. These are instrumented stream intervals, not exclusive
kernel compute or additive portions of the uninstrumented wall clock. The
sampled 99051/batch-200 event regime is especially perturbed: 26.13% overhead.
Profiler sessions take 2.32–4.67 times the matched plain session duration; never
use their session time as an implementation speed estimate.

After reconstructing device attribution from raw traces, the two MLP dW matrix
products take 1.50–2.21 ms per active native update: 7.66–8.56% of attributed
device duration. Their input shapes identify both MLP weight gradients; each
shape has 18 CPU matrix-multiplication calls (six blocks times three active
updates). Summing direct device events includes any associated memset kernels.
Sampling lowers this component to 0.47–0.77 ms, but adds 360 attributed device
events per update and increases forward work. The rest of backward propagation
still runs. Less sampled arithmetic therefore does not imply lower wall time.

The target's device-time share is small. Even its idealized elimination would
remove less than 9% of this attributed device-duration budget. This is **not a
rigorous wall-time speedup bound**: profiler perturbation, launch gaps, overlapping
activity, and changed kernel selection prevent that inference. It is sufficient
reason to deprioritize this narrow target on this workload. The result does not
rule out sampled gradients on different shapes, hardware or fused designs.

Native backward matrix multiplications, copy/fill/add operations, normalization
and activation/dropout gradients occupy much more time collectively. Attention
and tensor-layout execution are reasonable next targets to investigate, not
proven removable bottlenecks. The cost of individual Python, sampling, gather,
launch and layout components has not been causally isolated by this experiment.

## Attribution correction and remaining limits

The frozen summary collector follows CPU-parent ancestry. Autograd worker-thread
operators lack the main-thread phase range as a parent, so most backward device
work was classified as `outside`. The tiny original backward GPU self-time is
not a measurement of total backward cost. The integrity audit did not test this
semantic coverage. Preserve the original outputs and use the separate review.

The review joins device correlation IDs to host launch records and CPU external
IDs, then assigns the enclosing host phase across threads. It does not classify
asynchronous GPU work from the kernel's execution timestamp. Reconstructed sums
match the original profiler GPU totals for every operator in all 12 conditions
within 0.000001 microseconds. Only their phase assignment changes. Forward module
ranges remain forward-only; MLP dW identification uses checked operator shapes.

Trace boundary/auxiliary events without an active-phase launch are excluded and
counted explicitly in per-condition coverage files. This matches the saved
operator attribution budget, not every raw trace duration. GPU kernel availability
alone does not establish complete phase attribution. No full diagnostic rerun is
needed to recover these existing traces. Correct future phase collectors before
reusing this instrumentation. Existing frozen files remain unchanged.

## Recommended next study, one step at a time

1. Establish an efficient full-gradient reference. Keep historical FP32/math
   execution as a comparison and separately test supported optimized attention
   execution at fixed precision. Verify the backend actually selected; report
   unsupported configurations rather than silently enabling reduced precision
   or changing the deterministic policy.
2. If the trace suggests a specific layout rewrite, test one state-compatible
   attention/QKV layout change with identical parameters and full gradients.
   Validate forward outputs, every gradient and optimizer-state mapping against
   an explicit reference, with prospective numerical tolerances. Different
   kernels need not be bitwise identical. Preserve dropout semantics and label
   any reproducibility changes explicitly. Compare against the efficient native
   reference as well as the historical one.
3. Freeze an artificial cost screen only after correctness passes. Include setup,
   memory and synchronized uninstrumented timing, counterbalanced ordering and
   the existing practical saving target. No profiler-session speed comparison.
   Only a useful cost result justifies real-data noninferiority validation.

Reduced precision and compilation are separate possible later baseline controls;
do not mix them into an unexplained claimed algorithmic gain. Standard execution
improvements are baseline engineering, not a novel learning method. Reconsider
approximations only after the remaining cost budget is measured. Publication
stays on hold; the paper retains both the negative result and its mechanism.

PyTorch documents automatic SDPA backend selection, input restrictions and
possible numerical differences between implementations:
[SDPA, version 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).
Its profiler documentation describes collection options and shape-recording
overhead: [profiler, version 2.11](https://docs.pytorch.org/docs/2.11/profiler.html).
These references motivate validation; they do not establish a speedup here.
