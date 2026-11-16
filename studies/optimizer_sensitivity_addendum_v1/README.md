# Optimizer sensitivity — paper-support source addendum v1

Prepared and frozen 2026-10-27; full diagnostic pending. The source capsule
contains the matched-state runner, raw-tensor auditor, tests, CMD launcher,
prospective protocol/specification/freeze, required local source closure and
the pinned upstream attention source/license. Earlier packages are unchanged.

Original-workspace instructions: README_OPTIMIZER_SENSITIVITY.md. Provenance
retains absolute paths, so this is not a portable training release. Installed
torch/torchvision files are identified by EXTERNAL_DEPENDENCY_HASHES.json.
DATA_REQUIREMENTS.json identifies local archived checkpoints and training-data
dependencies; these are not copied here. Do not upload the whole project root.
No new data, training outputs or performance claims accompany this capsule.

Four CPU and four CUDA tests passed. Two tiny workflows audited, including
raw-gradient/update reconstruction, common state hashes, exact reference repeats,
and zero self-comparisons. CPU had 14 eligible replays; CUDA had 42. Artificial
trained fixtures exercise the code only. The two real archived checkpoints
and their 17600-step optimizer states were validated without model gradients.
Completed CUDA rerun skipped forward/backward and reconstructed saved statistics.

Six full-size cases will compare gradients at shared FP32 anchors and replay
AdamW from shared fresh/retained states. Source history uses common diagnostic
hyperparameters, not the original source schedule. Fixed gradient-magnitude bins
and coordinate traces test an optimizer-sensitivity hypothesis. There is no
accuracy/speed benchmark and no waiver of earlier BF16 gate failures. Raw tensors
remain internal until reviewed; no full diagnostic has been run or published.
SOURCE_PROVENANCE.json and SHA256SUMS.json identify this capsule's contents.
