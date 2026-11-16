# Stage13B preparation validation — 2026-10-23

No real Stage13B experiment or real-data scoring ran during preparation.
Existing checkpoints and traces were read for provenance; all executed updates
used artificial images and labels. Stage14 was not launched.

## Tests passed

Five CPU and five CUDA tests:

- Checkpoint-restored artificial continuation exactly reproduces an uninterrupted
  trajectory's per-step losses and final logits, with intermediate scoring.
- Reset preserves starting weights, BatchNorm buffers, RNG and scheduled rates;
  it removes classifier history. The first matched update changes the classifier
  differently while backbone updates remain identical.
- Audit reconstruction matches the exact crop/flip RNG consumption.
- Native ResNet checkpoint restoration reproduces the model state and optimizer
  ages of uninterrupted full training on artificial inputs.
- Reference qualification requires reproduction and both fixed-endpoint loss
  and accuracy improvement; artificial outcomes cannot qualify scientifically.

## Complete workflows

Final artificial outputs:

- `results/stage13B_sanity_reference_v1_final_cpu`
- `results/stage13B_sanity_reference_v1_final_gpu`

Both completed four trajectories /24 measured updates and reproduced their
known artificial reference traces. Integrity, batch coverage/order/hash checks,
augmentation RNG endpoints, source/final states, fixed BatchNorm and optimizer
age audits passed. Curves and endpoint statistics were independently computed
from saved logits. Their scientific qualification flags are deliberately false.

Completed CPU rerun verified existing results read-only. All35 output file
hashes were captured beforehand and are checked during finalization. No real
reference success is inferred from synthetic reproduction.

## Freeze and preservation

Freeze hash: `010e740c4ce5d1113c9dde28d33a605b3cf87a7ea434b4f4c2b17c210ee3064d`.
`FROZEN_REFERENCE_v1.json` binds33 scientific dependencies, the new specification,
original recipe/traces, source checkpoints, pretrained weights, split and runtime.
No source modification is allowed within this version after real execution.

`preparation_preservation_v1.json` protects1,215 prior files, including the
prepared Stage14 experiment and its final artificial outputs. Finalization
rechecks those files, both Stage13B synthetic completion manifests, CPU rerun
hashes, all frozen dependencies and source weights, and real-output absence.

New master v7.9 and paper memory v2 prioritize reference repair. Only the existing
CURRENT_RELEASE.md and README_STAGE9.md pointers change, with before/after
hashes. Earlier masters, memory v1, scientific protocols/results and manuscripts
remain unchanged. Stage14 stays prepared and queued.

Reference qualification is separate from audit integrity. After the user runs
Stage13B, review complete curves and reproduction checks before specifying any
local-learning comparison. No automatic hyperparameter tuning or next run.
