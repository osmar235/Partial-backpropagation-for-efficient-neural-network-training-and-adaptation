# Stage13C preparation validation — 2026-10-23

No real Stage13C training, calibration or development scoring ran during
preparation. Existing source/calibration files were read and hash verified.
All executed updates used artificial images and labels; Stage14 was not run.

## Six tests passed on CPU and CUDA

- Full training through the new local-model container exactly reproduces an
  uninterrupted artificial reference trajectory, including restored history.
- Classifier moments/ages and calibrated auxiliary moments map to the intended
  parameters. Active ages advance; inactive weights, moments and decay stay
  unchanged across multiple updates. BatchNorm buffers remain fixed.
- All four arms have identical initial final-block gradients.
- Auxiliary calibration leaves source weights unchanged and reproduces exactly
  when rerun with the same artificial batches and seeds.
- Review filters enforce quality, baseline readiness, charged cost and the
  best qualifying controls; artificial results cannot promote a candidate.
- Native ResNet tests using artificial inputs verify auxiliary moment tensors,
  exact original-versus-restored full model updates and identical initial
  full/local final-block updates with restored optimizer history.

## Complete synthetic workflows

Final outputs:

- `results/stage13C_sanity_trajectory_v1_final_cpu`
- `results/stage13C_sanity_trajectory_v1_final_gpu`

Each completes eight trajectories /48updates plus8auxiliary calibration steps.
Both integrity audits PASS. Both full artificial traces match all6expected
losses per seed. Calibration, state mapping, batch coverage/order/hash, RNG
endpoints, inactive parameters, finite gradients, optimizer ages, curve
arithmetic, timing/memory fields and charged-control comparisons are verified.

Scientific reference qualification is false in these random toy problems,
which do not improve both accuracy and loss. This does not affect trace/state
reproduction and provides no evidence about real efficacy or performance.
Candidate scientific promotion is explicitly disabled for synthetic outcomes.

Completed CPU rerun performed read-only auditing. All61 output file hashes were
captured beforehand and are rechecked at finalization. Real CUDA results are
still required; artificial timings and memory values are not scientific data.

## Freeze and preservation

Freeze hash: `a3b9e8272bc59aa9aafb95117e9b6539907da16df689d2b31e18149f3a6a2f23`.
`FROZEN_TRAJECTORY_v1.json` binds46 scientific dependencies, two source warmup
checkpoints, two prior auxiliary calibration checkpoints/streams, the original
recipe and reference traces, evidence, pretrained input, split and runtime.

`preparation_preservation_v1.json` protects1,281 prior immutable files. Finish
rechecks these, the frozen sources/evidence/checkpoints, both synthetic final
manifests, unchanged CPU output and absence of real Stage13C output. New master
v7.11 and README_STAGE13C.md describe the authorized experiment. Only the two
existing current pointers change, with before/after hashes recorded. All older
protocols, results, masters, paper memory and manuscripts stay unchanged.

The real run must reproduce auxiliary weights AND moments and qualify the full
reference before interpreting local efficacy. Review both final quality and
costs; do not infer useful learning from an integrity PASS or a memory-only
flag. No automatic continuation, tuning, extra epochs or publication action.
