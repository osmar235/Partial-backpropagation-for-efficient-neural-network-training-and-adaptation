# Stage14A preparation validation — 2026-10-23

No real Stage14 diagnostic, validation scoring or training was executed during
preparation. Existing source checkpoints were read for provenance only. New
checks used artificial inputs and randomly initialized networks.

## Artificial invariants

Five tests passed on CPU and the RTX 5080 with CUDA:

- An independently constructed token/attention path with unit masses matches
  native model logits and every parameter gradient within FP32 tolerances.
- Both mergers preserve CLS, total patch mass and the weighted sum of features;
  every source token receives a nonzero finite gradient.
- Proportional attention reproduces retained outputs when identical patch
  tokens are duplicated, including the separate CLS output. Unit masses agree
  with the native block.
- Native width192, six-block models backpropagate through both compressed arms
  with finite gradients, nonzero patch/positional gradients and unchanged state.
- Prospective filter boundary checks cover quality, time, memory, baseline
  readiness and forced exclusion of artificial outcomes from scientific passes.

## Complete synthetic workflow

Final outputs:

- `results/stage14_sanity_tokens_v1_final_cpu`
- `results/stage14_sanity_tokens_v1_final_gpu`

Each completed two artificial source states, three arms, two operation types,
three trials: 36 measured operations and 12 warmups, zero optimizer updates.
Both integrity audits PASS. Training streams, development labels, saved logits,
source state hashes, operation order, resource arithmetic and summary filters
were checked. Artificial quality/cost values have no scientific interpretation.

Completed CPU rerun performed read-only verification. All 21 output file hashes
are preserved in `cpu_before_resume_sha256_v1.json` and checked at finalization.
An earlier `prep_cpu` smoke is a retained development artifact and is superseded
by these final workflows.

## Freeze and preservation

Real freeze: `FROZEN_TOKEN_DIAGNOSTIC_v1.json`.
Freeze hash: `2a070bfee1b904b0cede831fa34365e3443769adcddff39764cba8bd581fec46`.
It binds 26 scientific source/spec/protocol files, primary evidence, the split,
two Stage11B full source checkpoint byte hashes, runtime versions and precision
settings. Changing scientific sources after the freeze requires a new version.

`preparation_preservation_v1.json` carries 1,155 verified prior immutable files.
The finish step rechecks them, both final synthetic completion manifests,
unchanged CPU outputs, real checkpoint hashes and absence of real Stage14 output.
Only CURRENT_RELEASE.md and README_STAGE9.md are intentionally updated existing
files; their before/after hashes are retained. Master v7.8 and the paper
discussion memory are new files. Prior studies and manuscript bytes are retained.

This validates implementation mechanics and provenance, not the real-data
accuracy or performance of intermediate compression. Execute the supplied CMD
launcher and review the results before any longer experiment.
