# One QKV split-layout change — 2026-10-25

The author approved the targeted attention/QKV layout study after the backend
screen failed. Compare historical math attention with one `qkv_unbind` adapter.
This is an exact-in-algebra implementation change, not a new gradient method.

## Single intervention

Installed PyTorch's packed self-attention projection forms a contiguous tensor
with shape [3, tokens, batch, width], then returns three independent indexed
views. Keep that projection GEMM, sequence-first ordering, transpose/reshape,
contiguous packed copy, attention views, math SDPA and output projection order.
Replace only `packed[0], packed[1], packed[2]` with `packed.unbind(0)`.

Hypothesis: a shared unbind backward can stack Q/K/V derivatives instead of
three independent select-backward scatter/zero operations followed by additions.
This does not remove the packed forward copy or prove a speedup. The narrow
adapter accepts only the existing batch-first self-attention call, no masks,
no causal flag, no attention weights output and zero attention dropout. Reject
unsupported calls instead of silently changing their behavior. All original
Parameter objects, state-dict names, optimizer mappings and outer dropout remain.

Both arms force SDPBackend.MATH. All parameters train. FP32, TF32/AMP/compilation
off, strict deterministic algorithms, same AdamW and dropout policy. No key-bias
exception, optimizer epsilon change, gradient modification or tolerance change.

## Correctness and mechanism before cost

Unit checks on CPU/CUDA: bitwise float64 attention forward comparison plus all
parameter/input derivatives, finite-difference gradcheck of input and every
attention parameter, preservation of an already-populated optimizer and checkpoint
mapping, full-model two-step comparisons, deliberate numerical-error rejection,
unsupported-call rejection, and actual split-graph/operator checks.

At the actual full-model batch shape, a single untimed CPU-profiler update must
show six math SDPA calls and no fused attention implementation. Count split
operators specifically on [3,65,batch,192] inside a forward-only range: native
must use 18 selects and zero unbinds; candidate six unbinds. Record any internal
forward selects inside unbind rather than mistaking them for independent
autograd nodes. Candidate must have six UnbindBackward0 nodes; reference zero,
and exactly 18 fewer SelectBackward0 nodes than the reference. Count backward
nodes across all threads, without assigning device time by parent-thread phase.
Forward split counts exclude select operations executed during backward.

Compare two updates with identical artificial inputs, initial weights and per-step
dropout seeds. Check every logit, loss, gradient, post-update parameter and Adam
first/second moment. Verify all gradients exist and are finite. RNG must match
the reference; repeating the same arm must reproduce model/optimizer hashes.
Preserve the preceding backend screen's elementwise tolerances exactly:

| Quantity | atol | rtol |
|---|---:|---:|
| Logits | 1e-4 | 3e-4 |
| Loss | 1e-5 | 1e-4 |
| Every gradient | 2e-5 | 5e-4 |
| Updated parameters | 2e-5 | 1e-5 |
| Adam first moment | 5e-6 | 1e-3 |
| Adam second moment | 1e-7 | 1e-3 |

Elementwise scaled error is abs(delta)/(atol+rtol*abs(reference)); require <=1
for every element. Failed numerical/repeatability gates exclude the candidate
from timing. Reference failure aborts. Do not change these gates after seeing
the full run. Existing key-bias sensitivity remains subject to the same rule.

## Prospective artificial cost screen

Two seeds 99081/99082, batches 256/200, six-block width-192 ViT, 65 tokens,
three heads, MLP width 768. Six counterbalanced rounds per condition, with
alternating/rotating available-arm order. Each trial recreates the identical
initial state, runs three discarded warmups, restores weights and empty AdamW,
then times the first update separately plus eight steady updates. Same historical
loss.item() synchronization; synchronized wall-clock boundaries. No profiler,
hooks or phase events inside timed updates. Model setup (including adapter
installation), warmup/restore, first-update costs and allocated memory are separate.

Primary rule: at least 10% median paired steady-update time saving versus native
math in all four conditions, after passing all correctness gates. Memory does
not substitute for time. At most 48 timing trials: 432 timed updates and 144
discarded warmups, plus 40 correctness updates and eight mechanism-probe updates
when both arms run all correctness checks. Actual counts are reported if a gate
blocks an arm. Timing repeats are not independent training replications.

No real data/checkpoint, accuracy claim, complete-adaptation claim or automatic
follow-up. A passing artificial cost result requires review and a separate
prospective real quality study. A failed screen closes this fixed implementation.

## Evidence and preparation

Freeze specification, sources, framework/device and execution policy. Hash
completed conditions and the root; preserve partial attempts and verify/skip
completed work on restart. The auditor resolves relative input paths before
checking absolute saved paths and verifies the root marker on read-only runs.
Reconstruct all raw paired ratios, numerical decisions, counts and dispatch checks.

Preparation uses a two-block width-24 model, batch three, seed99083, two timing
rounds, one warmup and two steady updates. CPU/CUDA tests and complete pipelines
must pass; candidate numerical failures block preparation readiness. Check both
relative/absolute audit paths, immutable rerun and tamper rejection/restoration.
Do not run the full-shape screen during preparation. Preserve all older evidence.

Implementation reference: installed torch.nn.functional._in_projection_packed
and multi_head_attention_forward, both covered by source hashes. The one changed
operation uses [torch.unbind](https://docs.pytorch.org/docs/2.11/generated/torch.unbind.html).
