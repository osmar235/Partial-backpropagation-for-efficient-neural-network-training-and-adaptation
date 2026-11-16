# Stage9E preparation validation — 2026-10-21

READY for the author-run GPU pilot. No real Stage9E output/training was created
during preparation. Protocol/spec/code/input-evidence hashes were prospectively
saved in `FROZEN_QUALITY_PILOT_v1.json` after the checks below.

## Mechanics and online allocation

Five CPU tests PASS in the existing project environment (Python3.14.3,
torch2.11+cu128). Coverage:

- Feasible nonadjacent-mask counts checked against exhaustive enumeration for
  windows0--8, all counts and both preceding-update states.
- Exact70/175 omission quotas under periodic, random and adversarial adaptive
  requests over repeated real-length epochs; no consecutive omissions.
- Current-feature exclusion from rolling thresholds, before history insertion.
- Both tiny and native six-block ViT: ordinary versus split full gradients,
  parameters and Adam state agree; predecided versus boundary partial paths
  agree. Prefix parameter/moment/counter state stays exactly fixed on partial
  updates. Nonzero prefix gradients prevent vacuous equivalence.
- Sustained quality target confirms only at the required final consecutive
  validation; first crossings and unattained/censored targets remain separate.

## End-to-end artificial workflows

CPU and RTX5080 CUDA final rehearsals each completed all TEN trajectories,
180 updates total (three epochs; tiny model;16 artificial training examples).
Batch3 creates five regular batches and one short batch each epoch, so full
handling of short batches and calibration exclusion are exercised.

Each rehearsal's independent auditor reconstructed quotas, random masks,
causal thresholds, LR/MixUp traces, targets and time summaries. It CPU-loaded
all ten checkpoints, verified finite tensors and moment/counter identities,
and compared final batch, augmentation, permutation, MixUp, CPU and CUDA dropout
streams across arms for each seed. All passed. Artificial primary target0 and
secondary target1 exercise target attainment and censoring; their apparent
accuracies/screen flags have no scientific meaning.

Final outputs:
`results/stage9_sanity_quality_pilot_v1_final_cpu/` and
`results/stage9_sanity_quality_pilot_v1_final_gpu/`.
An earlier successful all-regular-batch CPU rehearsal is retained separately
as `stage9_sanity_quality_pilot_v1_rehearsal_cpu`.

Completed CPU resume re-audited and skipped the pilot; all70 protected files
were unchanged by independent SHA-256 comparison. Before-resume manifest:
`quality_pilot_cpu_before_resume_sha256_v1.json`.

## Declared scientific execution

Two new development seeds1301/1302, five arms,100 epochs each, same cut4 and
prefix-freezing semantics. Real budget:176,000 total suffix updates across ten
runs;17,600 per arm. Each partial arm omits6,650 regular256-example prefix
updates (1,702,400 prefix examples). No partial-update accounting on the short
200-example batches. Full prefix counter17,600; skipping-arm prefix10,950;
all suffix counters17,600. Independent audit enforces these before completion.

Primary sustained80% target, three consecutive validations. Both-seed screen
requires >=5% shorter observed time versus full AND fastest reaching
predecided control, with final accuracy loss <=0.2pp against each. All missing
targets and failed controls remain visible. Two seeds are development evidence,
not a noninferiority/superiority test or publication result.

Hardware/setup warmup is separate; active-loop time and the primary observed
clock include actual calibration/controller costs. Observed target time also
includes validation, trace logging and scheduled telemetry. See the frozen
protocol for exclusions. No unmeasured diagnostic oracle is used online.

## Preservation and operating instructions

Prior results, executed sources, project masters and v7 manuscript are retained.
Preservation verification: `quality_pilot_preservation_v1.json`. Dated preparation
file hashes: `quality_pilot_preparation_sha256_v1.csv`. Only the current README
and release pointers are intentionally updated; older masters stay unchanged.

Run `scripts/run_stage9E_quality_pilot.cmd` with the existing project Python.
It runs mechanics tests, skips/rechecks the matching artificial CPU rehearsal,
then launches the frozen real study. Stop other GPU work. Allow approximately
0.5--1GB free, more for retained incomplete attempts. Timing depends on the
machine; about1--2 hours is a rough estimate from recent measurements.

After interruption rerun the SAME command: complete trajectories are hash
verified and skipped; an incomplete trajectory restarts in a NEW attempt.
Changed code/spec/environment fails explicitly. Send the entire
`results/stage9E_quality_pilot_v1/` folder on completion or any failure. No
automatic extension, new seed search, test evaluation or publication follows.
