# Optimized prefix/suffix cost review — 2026-10-28

Substantial training-block compute headroom remains under the qualified
BF16/automatic execution policy. This supports one bounded follow-up on quality
and complete adaptation cost, not a method-success or publication claim. The
precise savings and ordering of refresh schedules remain noisy. Full scientific
cost screen completed; no training, inference, backward or optimizer step during
this review. No next experiment prepared or launched.

## Integrity and provenance

The frozen auditor passes: 112 measured blocks, 10752 primary updates, 104 paired
comparisons, four fresh-process sessions, two source checkpoints, no retries.
Eight full/routed-full pairs have identical per-step losses, final model hashes
and final optimizer hashes. All fourteen arms' recorded learning trajectories
and final-state hashes also reproduce across all four timing sessions per source.
These repeats are not additional model seeds or new independent quality outcomes.

An independent standard-library audit verified frozen source/evidence/completion
hashes, 10752 step records, 1792 component records, all arm orders and refresh
counts, shared anchor/stream identities, partitions and all 104 paired contrasts.
It checks saved runtime assertions for finite states/gradients, expected optimizer
ages, missing inactive gradients and unchanged pure-suffix prefix weights. It
cannot independently reload those final states: the cost runner intentionally
did not archive every final checkpoint or gradient. No accuracy or validation
evaluation exists for these partial-update trajectories.

Warmup profiler records: 672 efficient SDPA forward calls, zero math/flash/cuDNN.
This documents first warmup calls only, not continuous training dispatch. Runtime:
RTX 5080, torch 2.11.0+cu128, CUDA 12.8. Pre/post-session telemetry is preserved;
it cannot rule out transient interference during every measured block.

## Complete set of paired training-block results

All candidates use the pinned reference attention, BF16 forward and explicit
FP32 loss. Parameters and moments are FP32. r0 means no prefix refresh; r4/r16
means full updates at zero-based steps 0, R, 2R, ... . Cuts freeze the stem,
class/position embeddings and first k blocks. Final norm and classifier always
train, even at cut 6 (only 2314 trainable coordinates on its suffix steps).

| Arm | Source | Median saving vs full | Range | Faster pairs | Median saving vs faster control* |
|---|---:|---:|---:|---:|---:|
| routed_full | 1501 | 2.73% | -6.91 to 20.66% | 2/4 | -2.05% |
| routed_full | 1502 | 0.76% | -7.95 to 9.67% | 2/4 | -1.22% |
| cut1_r0 | 1501 | 18.38% | 11.20 to 22.32% | 4/4 | 10.70% |
| cut1_r0 | 1502 | 7.89% | 2.25 to 17.84% | 4/4 | 6.10% |
| cut1_r4 | 1501 | 14.22% | 2.65 to 32.61% | 4/4 | 9.75% |
| cut1_r4 | 1502 | 11.67% | 4.26 to 19.71% | 4/4 | 7.76% |
| cut1_r16 | 1501 | 10.06% | 5.68 to 30.24% | 4/4 | 8.44% |
| cut1_r16 | 1502 | 8.92% | -3.46 to 12.70% | 3/4 | 2.83% |
| cut3_r0 | 1501 | 31.26% | 21.74 to 43.39% | 4/4 | 25.71% |
| cut3_r0 | 1502 | 30.38% | 22.49 to 37.88% | 4/4 | 27.96% |
| cut3_r4 | 1501 | 25.78% | 20.33 to 42.83% | 4/4 | 25.66% |
| cut3_r4 | 1502 | 17.01% | 14.76 to 21.45% | 4/4 | 13.84% |
| cut3_r16 | 1501 | 33.27% | 24.94 to 48.84% | 4/4 | 29.93% |
| cut3_r16 | 1502 | 33.75% | 28.55 to 37.42% | 4/4 | 30.64% |
| cut5_r0 | 1501 | 50.15% | 42.38 to 52.67% | 4/4 | 42.74% |
| cut5_r0 | 1502 | 48.02% | 45.33 to 53.69% | 4/4 | 48.02% |
| cut5_r4 | 1501 | 35.81% | 17.24 to 46.69% | 4/4 | 33.91% |
| cut5_r4 | 1502 | 38.11% | 29.59 to 44.20% | 4/4 | 34.94% |
| cut5_r16 | 1501 | 39.92% | 29.52 to 51.00% | 4/4 | 31.91% |
| cut5_r16 | 1502 | 45.43% | 35.69 to 52.59% | 4/4 | 43.60% |
| cut6_r0 | 1501 | 55.84% | 50.74 to 61.90% | 4/4 | 50.54% |
| cut6_r0 | 1502 | 52.29% | 46.86 to 57.44% | 4/4 | 49.94% |
| cut6_r4 | 1501 | 44.39% | 40.25 to 48.26% | 4/4 | 41.52% |
| cut6_r4 | 1502 | 46.78% | 43.93 to 48.82% | 4/4 | 45.22% |
| cut6_r16 | 1501 | 55.31% | 51.34 to 61.88% | 4/4 | 52.38% |
| cut6_r16 | 1502 | 57.39% | 53.77 to 59.38% | 4/4 | 55.19% |

*The faster-control comparison is a post hoc sensitivity check using the smaller
same-source/session full and routed-full wall times. It does not replace the
prospectively specified full comparator or introduce a new qualification gate.
All cuts 3/5/6, all three schedules, are faster than both controls in all sessions
(72/72 contrasts). Shallow cut 1 has some reversals, especially refresh every 16.

## What timing supports, and what it does not

The identical-learning routed-full control spans -7.95% to +20.66% savings versus
ordinary full; its median is only +2.73% / +0.76%, with 2/4 positive sessions in
each source. An extra routing branch has not demonstrated a speed improvement.
Short ~1–2 second blocks and machine/order effects create substantial spread.
Do not use nominal rankings or a few percentage points to choose an optimal cut
or refresh interval. Median paired percentages also need not equal ratios of
separate median times.

Cut3/r4: 25.78% / 17.01% median saving against full, 8/8 positive. Against the
faster control the medians are 25.66% / 13.84%, still 8/8 positive; the smallest
paired savings are 11.91% / 8.83%. Cut5/r4: 35.81% / 38.11% against full and
33.91% / 34.94% against the faster control. Cut6/r16: 55.31% / 57.39% against
full and 52.38% / 55.19% against the faster control. The last configuration omits
almost all parameter updates on 15/16 steps, so its quality is especially open.

Separate synchronized component probes support a computational explanation.
Median per-block mean backward costs for sources 1501/1502: full 9.05/9.50 ms;
cut3/r4 5.83/5.95 ms; cut5/r4 3.99/3.72 ms; cut6/r16 1.17/1.28 ms. Those probes
insert synchronization and cannot be summed/subtracted to explain the primary
wall-clock denominator. The trend is consistent with less backward work; it is
not a precise attribution of every millisecond saved.

Measured-minus-zero-extra-overhead-mixture residuals span -2.67 to +3.66 ms per
update; 41 of 64 are negative. Periodic refresh sometimes appears faster than
pure suffix-only even though it computes more work. Retain these observations
as timing/trajectory variability, not evidence of negative routing cost or an
optimal schedule. No inferential p-values, confidence bounds or new percentage
success cutoff are applied to this descriptive screen.

## Denominator and scientific limits

Primary wall time includes 96 updates with indexing, augmentation/MixUp, RNG,
routing, zero-grad, fresh full forwards, FP32 loss/readback, backward, AdamW and
in-memory trace collection. Data/model setup, calibration, profiled warmup,
restoration, component probes, telemetry/checks and artifact writing are excluded.
These savings apply to training blocks, not complete adaptation or time-to-quality.
The 16-update common calibration anchor is benchmark conditioning from trained
weights with fresh AdamW, not a source optimizer-history continuation. The full
cost screen also performed 128 anchor, 1792 warmup and 1792 component updates;
their time is outside the primary comparison.

No adaptive gate or approximation was evaluated, and no overhead for an unbuilt
decision rule is covered. Each refresh recomputes a real full gradient; suffix
steps omit prefix gradients, moments, counters and weight decay. Suffix-only
represents optimistic compute headroom, not a quality-preserving upper bound.
No generality across architectures/hardware or novelty established. Do not add
these percentages to the prior 43% BF16 adaptation saving or the scoped 6.34–6.40%
QKV saving. They have different clocks/comparators and may change quality.

## Historical context and next recommendation

Stage9C/D already showed cost headroom. Stage9E then failed the joint learning
screen: boundary-RMS and MixUp controllers were slower to quality and less
accurate than cheap blind schedules; per-epoch savings did not translate into
time-to-quality improvement. That negative branch remains closed. The present
optimized trained-checkpoint adaptation setting is different from Stage9E's
100-epoch training pilot. A new practical test here would assess that restricted
adaptation hypothesis; it does not rescue the old gate or demonstrate novelty.

Recommend one candidate, cut3/r4, as a conservative learning hypothesis: retain
the last three blocks, final norm and classifier updating every step and refresh
the prefix every four steps. Selection is post-screen scientific judgment,
not a preregistered winner or claim that it beats cut5/r4. Its cost advantage
survives both full controls while omitting less of the model than cuts 5 or 6.

The next prospective five-epoch adaptation comparison should contain full BF16/
automatic, permanently frozen cut3, and cut3/r4, starting from matched original
trained weights and common fresh AdamW under the established adaptation recipe.
Use the same fixed 0.30-point final-quality screen in common FP32 and own-policy
scoring, charge complete adaptation including setup/warmup/validation/artifacts,
and counterbalance timing sessions. Specify exact streams, ages, margins and
decision rules before any run. Do not carry a favorable cost-screen endpoint
or its 16-update calibration anchor into quality evaluation. The frozen-prefix
control tests whether periodic refresh adds useful quality compared with simply
training the suffix. If it does not, an adaptive gate has not earned a new study.
If this candidate fails, retain the negative result; no automatic boundary/R sweep.

This is a recommendation only. No next protocol, training script or run command
has been prepared. Publication remains on hold pending a distinct contribution.
