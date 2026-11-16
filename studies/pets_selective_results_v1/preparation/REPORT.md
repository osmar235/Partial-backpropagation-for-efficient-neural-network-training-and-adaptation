## Current action — Pet selective development prepared — 2026-10-29

The qualified baseline is preserved. The next executable comparison uses six
fresh development seeds (2701–2706), BF16 full backpropagation, frozen prefix and
periodic-r4 prefix updating. Prefix: stem + layer1 + layer2. Each of the six
possible arm orders occurs once. All arms share the 15-epoch, 690-step baseline
recipe, initialization and deterministic data/augmentation streams. Periodic
refreshes total 173; frozen prefix weights and AdamW state remain unchanged.
Full controls run in this new session for matched timings. No historical timing
substitution, boundary/cadence sweep or official-test evaluation is enabled.

Twelve functional checks (six CPU and six CUDA) passed, including exact full-arm
equivalence to the original update, native split-forward/suffix-gradient equality,
AdamW-state/weight-decay pause, frozen-prefix identity and tamper detection.
Twelve synthetic trajectories across CPU tiny and CUDA native-ResNet workflows
passed, with 48 exact checkpoint rescores including completed-run restart checks.
Real input/source/runtime preflight passed without a real training update.
The native smoke uses the real architecture and 224px shape at batch four;
batch-64 full-gradient capacity was already established by the real baseline.

Run scripts/run_pets_selective.cmd with the existing project Python environment.
Read README_PETS_SELECTIVE.md. Send results/pets_selective_development_v1 after
COMPLETE. No additional download is needed. Review the paired development
accuracy variability and timing feasibility before planning evaluation precision
(0.20 pp desired CI half-width, cap 24). Freeze the exact uncertainty adjustment,
seed count and inference before test access. Retain the 0.30 pp primary margin;
do not select evaluation based on a favorable candidate mean.

New standalone source/evidence capsule: paper_support/pets_selective_development_v1
(and ZIP). No real selective development or official-test result exists yet.
Manuscript v10 remains unchanged. Current master v7.58 / memory v47.
