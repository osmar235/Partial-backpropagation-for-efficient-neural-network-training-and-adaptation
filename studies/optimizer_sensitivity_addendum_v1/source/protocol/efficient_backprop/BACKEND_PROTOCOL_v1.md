# Efficient full-backpropagation: attention backend screen v1

Prepared 2026-10-25. Author approved testing efficient full backpropagation after
the runtime diagnostic. This first experiment changes only the attention backend.
Tensor-layout rewrites, reduced precision, compilation, optimizer changes and new
gradient approximations are deferred. It is baseline engineering, not a novel
learning-method claim. All inputs and initial weights are artificial.

## Fixed model and policy

Use the historical six-block ViT: width 192, three heads, MLP width 768,
patch size four, 65 tokens, dropout 0.1, attention dropout zero. Every parameter
trains with the same AdamW, learning rate 0.0001, weight decay 0.05 and loss
smoothing 0.1. FP32 throughout; AMP, TF32 and compilation off. Preserve strict
deterministic algorithms, deterministic cuDNN and historical cuBLAS workspace.

Seeds 99071/99072 and batches 256/200 form four conditions. Arms are forced MATH,
EFFICIENT_ATTENTION, FLASH_ATTENTION and CUDNN_ATTENTION via sdpa_kernel. Only one
backend is enabled inside each arm; math fallback is disabled for fused arms.
The context restores the original policy even on exception. Incompatibility with
this dtype/device/shape/determinism is a valid UNSUPPORTED outcome, not evidence
of a slow algorithm. Save warnings and the exact supported-category error.
Unexpected programming errors, OOM or numerical failures are not reclassified
as backend incompatibility. If all alternatives are unsupported, complete the
diagnostic and report no qualifying backend without changing precision/policy.

## Dispatch and numerical gate before timing

At the actual condition shape, one artificial update is profiled on CPU to verify
operator dispatch. Require the requested forward operator six times, no other
backend forward operator, and the corresponding fused backward operator six
times for fused arms. This probe is outside timing; it does not use phase-parent
attribution and does not inherit the previous profiler's cross-thread phase bug.
All parameter gradients must be present and finite.

Construct a math reference for two updates with matched inputs, initial weights
and per-update dropout seeds. Each supported arm runs the same two updates twice.
Compare every logit, loss, parameter gradient, post-update parameter and AdamW
moment against the reference. Parameter/state keys and shapes must agree. Check
optimizer step ages. RNG state must match math; repeated runs of the same arm
must have identical model/optimizer hashes and RNG. Thus different backend
arithmetic may differ within tolerance, while repeatability stays exact.

Elementwise acceptance is abs(candidate-reference) <= atol + rtol*abs(reference).
The JSON spec prospectively fixes these tolerances:

| Quantity | atol | rtol |
|---|---:|---:|
| Logits | 1e-4 | 3e-4 |
| Loss | 1e-5 | 1e-4 |
| Every gradient | 2e-5 | 5e-4 |
| Post-update parameters | 2e-5 | 1e-5 |
| Adam first moment | 5e-6 | 1e-3 |
| Adam second moment | 1e-7 | 1e-3 |

Save maximum absolute/scaled error for every tensor and step. Tolerance failure
or failed repeatability blocks that arm's timing, with a distinct failure status.
This small trajectory is an engineering check, not proof of long-run quality or
bitwise equivalence across backends. Do not relax tolerances after looking at
the full screen. Math correctness is mandatory; its failure aborts the run.

## Uninstrumented paired cost screen

Only validated arms enter six timing rounds per condition. Reverse base order
on alternating conditions and rotate available-arm order by round. All trials
reconstruct identical initial weights and empty AdamW state. Run three discarded
warmups, restore initial weights, recreate empty AdamW, then record the first
update separately and eight steady updates. Keep the historical loss.item()
behavior. Synchronize timing boundaries; no profiler, hooks or phase events are
present during timed updates. Backend context entry is outside these clocks.

Primary: median of the six per-round steady-time ratios to math. Require at least
10% saving in every seed/batch condition before recommending a real quality
study. An unsupported or correctness-failing condition disqualifies that backend.
Do not select a different winning backend for each condition. Report all arms,
all paired ratios (recoverable from trials.json), passing round counts, first
update and setup time, and absolute/incremental allocated memory. Setup covers
model construction, CPU snapshot, device transfer and optimizer creation;
warmup/restore time is separate. CUDA allocator reservations are also recorded.
The gate is steady operation cost, not complete adaptation or cold-start cost.

At most 96 trials: 864 timed updates plus 288 discarded warmups. Counts decrease
with unsupported/blocked backends. In addition there are 16 capability probe
attempts (unsupported attempts may stop partway) and 8 math-reference correctness
updates plus four correctness updates per supported condition/backend. Report
these counts separately; never treat timing rounds as training replications.

## Integrity and preparation

Freeze specification, sources, framework/environment/device and execution flags
before the user runs the full screen. Preflight GPU idle status and retain
before/after telemetry. Hash completion markers; reruns validate and skip complete
conditions. Preserve incomplete attempts. The independent auditor reconstructs
input hashes, backend eligibility, order, identity, counts and paired decisions.

Preparation uses four meaningful unit tests per CPU/CUDA device and one reduced
pipeline per device: two blocks, width 24, head width eight, batch three, one
seed, two rounds, one discarded warmup and two steady updates. Verify immutable
rerun and tamper rejection. Reduced-shape backend support and speed do not decide
full-shape eligibility. No full screen or real-data training during preparation.

Return results/efficient_backprop_v1 for review. No automatic follow-up. Preserve
all older frozen code/results. Correctness or cost success is not a novelty,
real accuracy, or publication-readiness result.

References: [SDPA backend context, PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.sdpa_kernel.html)
and [SDPA restrictions and numerical differences](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).
