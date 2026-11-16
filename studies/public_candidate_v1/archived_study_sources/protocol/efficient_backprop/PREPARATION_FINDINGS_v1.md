# Preparation findings — 2026-10-25

The executable screen and audit pipeline are ready. Four CPU and four CUDA unit
tests pass, as do complete reduced CPU/CUDA pipelines, read-only reconstruction,
immutable replay and tamper rejection/restoration. These are engineering checks;
the small CUDA run **does not qualify efficient attention**.

At two blocks, width 24, batch three:

- Math dispatch and numerical checks pass.
- Efficient attention dispatches both the requested forward and backward kernels
  with strict determinism. Repeatability and dropout RNG equality pass. All
  logits, losses, gradients and AdamW moment tensors pass their tolerances.
  Two post-update checks fail on block zero's packed attention projection bias.
- Flash reports this torch build was not compiled with flash attention.
- cuDNN attention reports incompatibility with strict determinism.

An additional four untimed artificial updates isolate the failed parameter
differences to the **key-bias slice**: maximum 4.28436e-5 after step one and
6.66806e-5 after step two. Its gradient differences are approximately 6.52e-9 and
7.68e-9. Query/value bias differences are much smaller. Overall maximum logit
difference is 2.08616e-7 and gradient difference 8.64267e-7.

Interpretation: a shared key bias adds the same query-dependent scalar to every
attention logit in a row, which cancels under softmax in exact arithmetic. Tiny
floating-point residual gradients can still produce visible AdamW updates in
this redundant direction because its default epsilon is 1e-8. This explains why
small functional differences and larger bias-coordinate differences can coexist;
it does not establish long-run numerical or quality equivalence.

The existing parameter gate remains binding. No tolerances, gradients, optimizer
epsilon, RNG, backend policy or model parameters were changed to make this pass.
The frozen full-shape run rechecks the actual six-block, width-192 model and
batches 256/200. If the mismatch persists, that arm's timings are skipped and it
cannot qualify. Do not report a candidate speedup from blocked or smoke results.

If no full-shape alternative passes, review these records before any amendment.
A future functional-equivalence or gauge-aware criterion would require a new
explicit protocol and independent validation; it is not part of this screen.
No full-shape timing or real-data training was performed during preparation.
