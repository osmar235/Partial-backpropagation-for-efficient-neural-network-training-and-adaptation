# Stage10A preparation validation — 2026-10-21

READY for the author-run representation screen. No real Stage10A training or
output was created during preparation. Source/spec/protocol/input-evidence
hashes were prospectively frozen in `FROZEN_REPRESENTATION_v1.json`.

## Mechanics

Six tests passed on CPU and CUDA in the existing Python3.14.3 / torch2.11+cu128
environment. They verify Haar inverse reconstruction, energy preservation,
linearity under MixUp, exact coarse-average identity and transform gradients;
lossless pixel grouping; shared input16 body initialization; identical
dual-branch trainable shapes/capacity/initialization; a nonvacuous detail effect
on predictions with identical coarse inputs; full parameter/moment updates;
and target confirmation/censoring.

Repeated same-model training was BITWISE equal for losses, final parameters
and Adam state on both CPU and CUDA. Replay covers all five tiny architectures
and a native six-block Haar model, with nonzero detail/backbone gradients after
initial zero-head updates. Strict deterministic flags are enforced as errors,
including cuDNN determinism and math-only attention. No silent fallback.

This establishes reproducibility for the tested operations/environment, not
cross-device, CPU-versus-CUDA or all-release numerical equivalence. Different
model shapes do not share identical dropout masks; compatible input16 bodies
do share draws and are audited accordingly.

## Complete artificial workflows and replay

Final CPU and RTX5080 CUDA workflows each completed TEN trajectories and180
updates on artificial data, tiny model widths and three epochs. Batch3 gives
five regular batches and a final single example, exercising short-batch handling.
The independent audit reconstructed LR/MixUp/seed traces, input counts, clocks,
targets and flags; loaded all ten checkpoints; verified all parameter counters
and finite moments; and matched data RNG streams across all arms and dropout
streams within compatible input16 bodies. All passed.

Final outputs:
`results/stage10_sanity_representation_v1_final_cpu/` and
`results/stage10_sanity_representation_v1_final_gpu/`.
Artificial target0/1 and all apparent accuracies/flags have no scientific meaning.

A separate full CPU replay matched all TEN step traces and all validation
learning curves exactly. Its1,416 serialized checkpoint tensor payloads matched
byte for byte, including parameters, optimizer states and RNG tensors. Proof:
`replay_verification_v1.json`; output `stage10_sanity_representation_v1_replay_cpu`.
An earlier CPU rehearsal is retained separately.

Completed CPU resume re-audited/skipped without changing any of70 files.
`cpu_before_resume_sha256_v1.json` supplies the before-state hashes.

## Scientific design and costs

Two new DEVELOPMENT seeds1401/1402, five arms,100 epochs each. Every parameter
receives17,600 updates;176,000 updates total. All use standard AdamW and one
joint classification loss. No local auxiliary objectives, evolutionary search,
random optimizer or selective-gradient skipping is being tested yet.

The expensive branch sees32x32 (65 tokens) for full/small RGB,16x16 (17 tokens)
for coarse/pixel/Haar. The dual representations retain all source coefficients;
low token count does not prove lower end-to-end cost. Transforms and both
branches are evaluated inside the timed training/evaluation loops. Report
actual parameter counts, wall time, target attainment and final accuracy.
The capacity-matched pixel control distinguishes structured band allocation
from the mere presence of a second branch; other representation/conditioning
effects remain part of that comparison.

Primary practical screen: sustained80% over three validations, >=5% faster
observed time than full AND best reaching ordinary baseline, final accuracy
within0.2pp of best ordinary final accuracy, BOTH seeds. Additional flags require
>=0.5pp endpoint gain over coarse and >=0.2pp over pixel with <=1.05 active-
training time ratio to pixel, both seeds. No thresholds/duration changes after
outcomes. Two seeds and reused validation are development evidence, not a test
of broad superiority. Small32x32 images are a bounded initial workload only.

The full baseline is rerun under the NEW deterministic settings; old Stage9E
timings are not substituted. The300epoch LR horizon is retained; only the first
100epochs are trained. No new recipe search or hidden data preprocessing cost.

## Operating and preservation

Prior science, failed pilots, frozen sources, masters and v7 manuscript are
preserved; publication stays on hold. See `preservation_check_v1.json` and
`preparation_sha256_v1.csv`. Current project master:v6.9.

Run `scripts/run_stage10A_representation.cmd` using the existing project Python.
It checks CPU/CUDA mechanics, verifies/skips the matching artificial CPU run,
then executes the frozen scientific comparison. Stop other GPU jobs first.
Reserve around0.5--1GB, more for retained incomplete attempts. Ten100epoch
trajectories are planned; actual runtime depends on the new model/kernel costs.

Output `results/stage10A_representation_v1/`. Send the whole directory on
completion or failure. Rerun the same command after interruption: complete
trajectories verify/skip; incomplete trajectories restart from initialization
in new retained attempts. Changed source/spec/environment fails explicitly.
No automatic test evaluation, architecture/optimizer search or publication.
