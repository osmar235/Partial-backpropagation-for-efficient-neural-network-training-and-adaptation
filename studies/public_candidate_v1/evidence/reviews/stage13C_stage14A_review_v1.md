# Stage13C and Stage14A independent review

Reviewed 2026-10-24. Both completed runs pass the integrity audit. Neither
candidate passes its frozen scientific acceptance rule. Stage13C now supplies
a valid negative comparison against a working full-training reference;
Stage14A supplies evidence of a useful compute/accuracy tradeoff, with an
accuracy shortfall that still needs to be resolved.

## Stage13C: the local recipe fails at the fixed endpoint

All eight trajectories completed the predetermined 352 updates, with matched
source states, batches, augmentations, schedule and optimizer-state mapping.
The auxiliary heads received 128 calibration updates per seed. Calibration
cost is charged in full to each arm that uses it. The two full trajectories
reproduce all 704 original reference training losses and qualify in both seeds.
The first-step deterioration does not invalidate this multi-step comparison.

| Arm | Accuracy, seed 1601 | Accuracy, seed 1602 | Charged time, seed 1601 | Charged time, seed 1602 |
|---|---:|---:|---:|---:|
| Initial source | 81.34% | 81.08% | — | — |
| Full | 88.58% | 89.74% | 11.942 s | 11.937 s |
| Connected auxiliary (`full_aux`) | 86.54% | 87.04% | 13.517 s | 13.069 s |
| Detached local (`local`) | 80.32% | 77.56% | 12.931 s | 12.616 s |
| Suffix only | 86.10% | 87.06% | 6.244 s | 6.226 s |

Local training finishes 8.26 / 12.18 percentage points below full training and
1.02 / 3.52 points below its own starting accuracy. Its endpoint cross-entropy
also worsens relative to its source: 0.555629 -> 0.576985 and
0.557440 -> 0.687694. It recovers partially after a large initial drop but
does not recover its starting quality within the frozen budget.

With calibration included, local training is 8.28% / 5.69% slower than full.
Its maximum incremental operation allocation is only 6.38% lower, failing the
20% memory criterion. Excluding calibration, its raw training-time savings
are only about 2.8% / 2.4%, still below the 10% criterion. Charged time includes
data transformation and optimizer updates; it excludes evaluation, restoration,
audits and logging. These are scoped training costs, not complete job wall time.

Connected auxiliary training also underperforms full training by 2.04 / 2.70
points. Detaching further lowers accuracy by 6.22 / 9.48 points relative to
that matched connected control. Suffix-only training saves about 48% of charged
time but loses 2.48 / 2.68 points, outside the frozen 0.30-point margin. Full is
the only quality-qualified control in either seed. No candidate promotion.

**Decision:** park this local-loss recipe. Preserve its negative result and
the reference-repair lesson. Do not extend training or change its threshold
retroactively. This experiment does not disprove local learning in general
and is not a replication or refutation of published methods.

![Fixed trajectories](local_trajectory.png)

## Stage14A / idea 4: savings are real, quality criterion fails

The diagnostic reduces 65 tokens to 33 after block 3 of a six-block ViT. It
compares full tokens, spatial averaging and similarity-based merging at the
same two frozen, fully trained checkpoints. CLS and token mass are preserved.
There are no optimizer updates or adaptation training in this diagnostic.

| Seed | Mode | Accuracy | Change vs full | Forward/backward time saved | Incremental memory saved |
|---|---|---:|---:|---:|---:|
| 1501 | Full | 82.46% | — | — | — |
| 1501 | Spatial | 81.32% | -1.14 pp | 20.66% | 26.34% |
| 1501 | Similarity | 82.16% | -0.30 pp | 21.52% | 26.39% |
| 1502 | Full | 82.74% | — | — | — |
| 1502 | Spatial | 81.00% | -1.74 pp | 23.41% | 26.33% |
| 1502 | Similarity | 82.14% | -0.60 pp | 23.52% | 26.39% |

Similarity merging retains 0.84 / 1.14 points more accuracy than spatial
averaging at comparable measured costs. The 0.30-point loss in seed 1501 is
exactly the inclusive frozen limit (15 net fewer correct predictions among
5,000 examples); that seed passes. Seed 1502 loses 0.60 points (30 net fewer
correct) and fails. Both seeds must pass, so similarity and spatial merging
both fail the predeclared screen. The threshold remains unchanged.

The time column measures forward plus backward operations, excluding optimizer
steps and common input preparation. It is not an observed end-to-end training
speedup. Memory is incremental allocated memory above the pre-operation
baseline, not total GPU memory. Similarity reduces absolute peak allocated
memory by approximately 18.65% in each seed. Inference operation time falls
20.82% / 22.42%; this is a secondary descriptive measurement. Timing uses the
frozen repeated-batch design on one GPU; tiny differences between merging
modes do not establish a reliable speed advantage.

**Decision:** prioritize a separately specified, bounded accuracy-recovery
experiment for similarity merging. Hold the compression boundary and size
fixed, compare a short matched fine-tuning budget against full and spatial
controls, and count all recovery costs. Predetermine endpoints and stopping
rules before running. A training-efficiency claim would additionally require
time-to-matched-quality accounting for the already trained source and all
adaptation; this diagnostic alone cannot establish that claim.

This follow-up is a recommendation, not a prepared protocol or executed run.
Stage14A remains a failed screen even if a future recovery study succeeds.
The existing protocol identifies the mechanism as ToMe-inspired; generic
token merging is not a new algorithm, and this is not an exact reproduction
of that work. Any eventual contribution must be established beyond this
diagnostic tradeoff and compared with appropriate prior methods.

![Token tradeoff](token_tradeoff.png)

## Independent verification and scope

- Frozen auditors pass: 2,816 trajectory updates plus 256 calibration updates
  for Stage13C; 144 measured operations plus 36 warmups for Stage14A.
- Independently recomputed accuracy and float64 cross-entropy from all 56
  Stage13C curve points and all six Stage14A mode/seed prediction sets.
- Recomputed timing, memory ratios and acceptance decisions from raw records.
- Reloaded Stage13C's two source and eight final models and rescored all six
  Stage14A mode/seed combinations. All 16 scoring conditions reproduce saved
  logits exactly (maximum absolute error 0); model states remain unchanged.
- Verified hashes for all 1,373 protected prior and raw-result files. No
  optimizer updates, test-set scoring, threshold changes or manuscript edits
  were performed during review. Only the two current project pointers change;
  their before/after hashes are recorded separately.

These are exploratory results on previously used development data, with two
seeds per study. The different architectures and source checkpoints prevent
a direct causal comparison across Stage13C and Stage14A. The result does not
establish generalization to other tasks, statistical equivalence, or a new
publishable method. Publication remains on hold pending a convincing useful
contribution and subsequent independent confirmation.

Supporting files: `independent_audit.json`, the two frozen audit rechecks,
`stage13C_independent_curves.csv`, `stage13C_independent_summary.csv`,
`stage14A_independent_summary.csv`, `saved_state_rescore.csv`, and the
preservation/seal manifests in this directory.
