# Stage11B: fixed midpoint growth pilot

Prepared 2026-10-22, after Stage11A and before real pilot outcomes. The author
approved exploring idea 1, small full-resolution training followed by growth.
Stage11A passed mechanics and step-cost feasibility. It did not establish
trajectory efficiency. This pilot asks whether later growth can recover quality
without consuming the early saving. Generic function-preserving growth is prior
art; no novelty or submission claim follows from this pilot.

## Fixed design

Two fresh real-development seeds, 1501 and 1502, use the original training-only
CIFAR-10 split: 45,000 train / 5,000 validation, no test-set access. Five arms,
each 100 epochs / 17,600 updates. Execution order is reversed between seeds.

| Arm | Epochs 1–50 | Epochs 51–100 | Adam at transition |
|---|---|---|---|
| full | width192 | width192 | continued |
| small | width96 | width96 | continued |
| full_reset | width192 | width192 | fresh |
| small_reset | width96 | width96 | fresh |
| grow | width96 | width192 | fresh |

All start from scratch at full 32x32 input resolution, patch4, 65 tokens, six
blocks and three heads. Expansion uses the frozen Stage11A map, including Q/K
scaling, duplicated residual channels and split outgoing weights. Expansion
preserves evaluation logits within atol5e-5 / rtol1e-5, not stochastic dropout
or optimizer trajectories. Duplicated channels can subsequently diverge.

Identical seed groups must have bitwise-identical initial states, pre-transition
loss/validation histories, weights and Adam moments. Data RNG streams must match
across all arms throughout. Strict deterministic FP32/math attention is inherited
from Stage10. This is a controlled mechanism pilot; optimized-baseline and
generalization comparisons remain necessary for any eventual efficiency claim.

Batch256, AdamW lr.001/wd.05, MixUp.2, label smoothing.1; same 300-epoch cosine
schedule horizon, initial five-epoch LR warmup, minimum ratio.01. The schedule
continues on the global update index after epoch50; no LR restart or repeated
warmup. Sixteen discarded hardware-warmup steps precede restored initial states
and fresh Adam in every arm. There is no excluded warmup after expansion.
All midpoint checkpoint/check overhead and the first cold wide update count.
Numeric settings and orders are authoritative in growth_pilot_spec_v1.json.

## Prospective primary endpoint and stop rule

Primary: complete-run wall time at comparable final epoch100 accuracy. This is
a new prospective pilot endpoint: a low time-to-target threshold can be reached
before expansion and cannot establish successful later recovery. Earlier stage
criteria/results remain unchanged. No best-epoch selection or adaptive stopping.

For each seed, find the best final accuracy among all four ordinary controls.
Quality-qualified controls are within 0.2 percentage points of that best. Growth
must finish within the same 0.2-point margin and use at least 5% less complete-run
time than BOTH uninterrupted full and the fastest quality-qualified control.
Thus a competitive fast small model cannot be ignored, nor can a reset control.
Full and growth must each sustain 80% validation for three consecutive epochs.
All conditions must hold in both seeds to call this a promising development
hypothesis. Two reused-validation development seeds provide no significance,
held-out generalization or novelty evidence. A failed full readiness check makes
the efficacy screen inconclusive; do not extend epochs or lower the target.

Report all five arms, final quality, charged times and individual seed flags.
Report sustained80% and82% crossings descriptively with non-crossings censored
at100. Reset contrasts diagnose whether an apparent benefit is explainable by
restarting Adam. No automatic transition search, seed expansion or new method.
Review the completed evidence before choosing further work.

## Timing and provenance

Complete-run clock starts before model construction and ends after final
checkpoint serialization and CUDA synchronization. It includes construction,
warmup, state restoration, data-RNG preparation, training, per-epoch validation,
trace writes, telemetry, printing, midpoint before/after checkpoint IO, function
checks and growth/reset. It excludes common once-per-launch dataset loading,
post-run summary/hash sealing and the final independent audit. Dataset loading
is logged separately. This is a standardized instrumented research workload;
validation/checkpoint overhead limits transfer to other deployment schedules.

Training, validation, hardware-warmup, transition and final-save times are also
reported separately. Every arm saves midpoint states and executes the same
evaluation-equivalence probe on eight training images (without updates).

Freeze all scientific sources, protocol, spec, tests, launcher and supporting
Stage11A evidence before real output exists. Record environment/split settings,
per-step losses/LR/MixUp/dropout seeds/width/Adam age, per-epoch metrics, midpoint
and final weights/Adam/data RNG states. Independent audit reconstructs update
schedules, exact growth map, reset/continued counters, matched histories and
decision arithmetic. A successful integrity audit is distinct from efficacy.
Completed runs are immutable; retry skips completed runs and starts a new
attempt for an incomplete arm. Completed-study resume is read-only verification.
Artificial CPU/CUDA rehearsals test mechanics, never scientific performance.
