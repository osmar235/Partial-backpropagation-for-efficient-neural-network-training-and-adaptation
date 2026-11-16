# Stage 8 preparation checks — 2026-10-18

All checks below passed using project Python 3.14.3, PyTorch 2.11.0+cu128 and
torchvision 0.26.0+cu128. **No real CIFAR Stage 8 training or new evaluation seed
was executed.** Native transformer forward/gradient tests and the full tiny
workflow rehearsal used CPU and artificial data. No GPU performance result
is claimed. Production uses CUDA and remains to be tested through development.

`scripts/test_stage8_vit.py` passed nine tests:

1. Native six-block transformer output shape, complete structural partition,
   native LayerNorm and exact suffix-gradient agreement with full autograd.
2. AdamW explicit-zero prefix transition agrees exactly with the independent
   full-gradient/zero-prefix reference, including continuing step counters.
3. Replay repeats the last full displacement and preserves all prefix AdamW
   state tensors on a partial step.
4. Training-only stratified 45k/5k split is disjoint, class-balanced and cannot
   construct the test dataset during development (mock rejects `train=False`).
5. LR warm-up/cosine endpoints, exact periodic counts, no adjacency and all
   pairwise evaluation ordering/early-late counts.
6. Identical resume skips completion, changed LR is rejected, and the saved
   model/optimizer checkpoint can be loaded using tensor-only safe loading.
7. Evaluation refuses to load data without a prior freeze.
8. An inadequate baseline fails the readiness guard and records that failure.
9. Busy-GPU utilization triggers the preflight guard.

`scripts/run_stage8_00_sanity.cmd` then completed an entire artificial workflow:
four baseline screens, two calibration runs, a synthetic-only freeze, all 24
evaluation arms, paired reports and targets. Synthetic seeds 9001/9002 and
9101–9106 are distinct from reserved scientific seeds. Its independent integrity
audit verified all 192 artificial evaluation updates, gate/EMA decisions, masks,
LRs, timing reconciliation, target crossings, identical initialization within
seeds, checkpoint hashes and every prefix/suffix AdamW step counter.
Outputs: `results/stage8_sanity_v2/`; these are diagnostic records, never paper
accuracy/speed evidence. The earlier development rehearsal is preserved in
`stage8_sanity_v1/` as implementation history.

The preparation telemetry query succeeded on the RTX 5080 and recorded high
utilization from existing work. Scientific profiling/training was not started
against that load. Per-epoch telemetry records driver, clocks, power, temperature,
utilization and pstate, outside the training timer.

The final preparation pass also checks previously frozen files and records
local source hashes in `stage8_preparation_sha256.csv`. All old Stage 6/7
scientific outputs, original release and v6 manuscript remain unchanged.

These tests establish implementation consistency, not transformer baseline
quality, accuracy preservation, timing advantage, sufficient statistical power,
or acceptance by a journal. Development readiness and the new fixed evaluation
must supply that evidence. A failed/inconclusive result is retained.
