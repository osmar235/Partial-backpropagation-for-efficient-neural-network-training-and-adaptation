# Stage10A completed review - 2026-10-21

The tested Haar coarse/detail design failed all three frozen development flags
in both seeds. The full baselines reached the primary target, so this is a
candidate failure rather than a baseline-readiness failure. Publication remains
on hold. Do not extend training or introduce optimizer mixtures to rescue this
configuration automatically.

## Outcomes

Final validation accuracy at the fixed 100-epoch endpoint:

| Model | Parameters | Seed 1401 | Seed 1402 | Mean | Sustained 80% reached |
|---|---:|---:|---:|---:|---:|
| Full RGB32 | 2,693,578 | 82.04% | 82.26% | 82.15% | 2/2 |
| Narrow RGB32 | 683,242 | 81.92% | 80.94% | 81.43% | 2/2 |
| Average-pooled RGB16 | 2,684,362 | 74.50% | 75.42% | 74.96% | 0/2 |
| Pixel dual branch | 2,697,322 | 74.78% | 74.68% | 74.73% | 0/2 |
| Haar dual branch | 2,697,322 | 74.98% | 74.70% | 74.84% | 0/2 |

The Haar model finished 7.06 / 7.56 percentage points below full, averaging
7.31 points below. All low-resolution arms remain censored at 100 epochs for
the primary sustained target. A single crossing is insufficient: the protocol
requires three consecutive validations. No arm sustained the secondary 82%
target; it is not substituted for the primary outcome.

| Frozen Haar check | Seed 1401 | Seed 1402 | Result |
|---|---:|---:|---|
| Practical target and quality | No sustained target; -7.06 pp | No sustained target; -7.56 pp | Failed both |
| Detail gain over coarse; requires >=0.50 pp | +0.48 pp | -0.72 pp | Failed both |
| Gain over pixel; requires >=0.20 pp | +0.20 pp | +0.02 pp | Insufficient jointly with cost |
| Training cost / pixel; requires <=1.05 | 1.06464 | 1.03448 | Seed 1401 fails |
| Joint structure flag | Failed cost | Failed accuracy | Failed both |

Mean Haar-minus-pixel endpoint gain is only 0.11 pp. The mean of paired active
training cost ratios is 1.04956 (about 5% extra); this average does not override
the frozen per-seed requirements. A Haar time-to-target ratio is undefined
because the target was never sustained; censored runs are not assigned a
successful target time.

## What the ordinary control tells us

The narrow full-resolution model reached sustained 80% in 203.26 / 230.20
observed seconds, versus full 339.32 / 293.68 seconds: 40.10% / 21.62% less.
Its final losses relative to full were 0.12 / 1.32 pp. Thus it does not meet
the desired 0.20-pp quality tolerance in both seeds and is not a useful-method
success under our joint criterion. It is also an ordinary architecture control,
not a novel contribution.

Within this workload, retaining full spatial resolution with a narrower model
was much better than the three tested low-resolution allocations. This is a
design clue, not a causal isolation of token count, bandwidth or optimization.
The Haar transform retains all coefficients, but the learned branches and
fusion may fail to exploit them. Lossless input coordinates do not guarantee
that a particular network can use the information efficiently. This review
does not establish why the allocation failed.

## Verification and preservation

- Re-ran the frozen read-only audit: all ten runs, 176,000 updates, initialization
  hashes, source/spec/evidence hashes, matched data streams, compatible dropout
  streams, trace arithmetic and every parameter's Adam state passed.
- Independently reconstructed primary/secondary crossings, confirmation epochs,
  clocks, censoring and all candidate flags from raw epoch traces. All agree
  with the original completed output. The frozen audit result also matches.
- Loaded all ten saved final checkpoints and re-evaluated the same training-only
  5,000-example validation split on the RTX 5080 with the frozen strict settings.
  Every final accuracy reproduced exactly; every validation loss agrees within
  1e-6. No training or test-dataset evaluation occurred during review.
- Verified 754 immutable files, including all completed raw Stage10A files,
  executed sources, preparation evidence, previous results/masters and preserved
  manuscript v7. The two current project pointers are separately hash tracked
  before and after their intentional status update. Old manifest entries remain
  historical; the original manifests and frozen sources are unchanged.

Audit PASS describes integrity, not candidate efficacy. This is a two-seed
development screen on reused validation, one 32x32 CIFAR-10 recipe and a fixed
100-epoch budget within a 300-epoch schedule. Architecture-specific tuning was
not performed. Neither statistical superiority nor a universal failure of
wavelet/frequency approaches follows. Historical stages used different
determinism settings; their wall times are not direct comparison baselines.

## Decision

Close this tested coarse/detail design. Keep the negative evidence. The results
do not supply the representation advantage that was the prerequisite for
adding local objectives, random optimization or genetic search. Another
optimizer would also leave the allocation mechanism untested as an explanation.

Return to design before spending another training budget. Any new experiment
should state which information or interaction the model must preserve, why it
should beat a narrow full-resolution control, and what observation would reject
that mechanism. No new architecture, schedule, seed sweep or training command
is created by this review. Prior Stage9E failure and the publication hold remain.

Evidence: `independent_audit.json`, `frozen_audit_recheck.json`,
`independent_seed_results.csv`, `independent_mean_results.csv`,
`independent_targets.csv`, `independent_screen.csv`,
`checkpoint_validation_rescore.csv`, `rescore_metadata.json` and
`preserved_inputs_sha256.csv`. Reproduction helper:
`scripts/review_stage10A_results.py --rescore-cuda` (review only).
