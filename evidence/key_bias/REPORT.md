# Key-bias functional-impact review — 2026-10-27

The completed diagnostic passes integrity checks. Key-bias-only update changes
have near-zero effects under FP32 evaluation in this setup, but are not
functionally inert under BF16. Restoring reference key bias does not reliably
remove the BF16-derived full-update output discrepancy. These results qualify
the earlier parameter-space finding; they do not establish a practical fix,
long-run quality equivalence, speed improvement or a novel optimization method.

## Integrity and interpretation scope

Four unfragmented anchor records from two trained models, eight optimizer-state
conditions, 40 component cases, 840 finite forward outputs, 240 within-policy
comparisons and 48 cross-policy comparisons. All 48 reference repeats are
bitwise exact, including automatic BF16. Both trained sources share the original
256-example training batch and a disjoint second 256-example training batch.
Neither is a validation/test benchmark; the second batch was excluded from the
diagnostic gradient batch, not from source training.

The frozen auditor passed after 48 saved-gradient optimizer replays on cloned
parameters. It recomputed donor masks/hashes and verified original deltas and
optimizer states. A separate CPU implementation independently reconstructed
loss with log-sum-exp, vector norms, prediction changes, interaction terms and
all stored comparisons from saved logits. Explicit FP32 loss reconstruction
agrees with FP64 within 9.35e-8 absolute error. No model forward/backward or new
learning trajectory was run during review. The independent audit ran no
optimizer steps; the frozen auditor's 48 replays are reported separately.
All source/evidence/completion hashes pass, with prior records preserved.

## Main functional result

The following rows include all four BF16 gradient origins, both models, both
anchors/history conditions and both batches. The forward implementation is
always the pinned reference; only its evaluation precision/backend varies.

| Evaluation policy | Comparisons | Key-only max logit change | Key-only max absolute loss change | Comparisons with prediction changes |
|---|---:|---:|---:|---:|
| reference_fp32_math | 64 | 3.8146973e-06 | 3.486423e-08 | 0/64 |
| reference_bf16_math | 64 | 0.03125 | 0.00040684636 | 6/64 |
| reference_bf16_auto | 64 | 0.03125 | 0.0004158667 | 8/64 |

Under FP32, key-only RMS logit change is 2.13e-7–2.84e-7 and no predictions
change. For archived optimizer history specifically, max logit change is
3.10e-6 and max absolute loss change 1.92e-8. This supports a local distinction
between large parameter discrepancy and very small functional effects when
evaluation stays FP32. It is not a universal tolerance or a quality proof.

Under BF16, key-only RMS logit change is 0.00242–0.00331. Its norm is often
similar to the full-update discrepancy (median key/full ratio 0.913 for math
and 0.956 for automatic execution). Those are ratios of norms, not percentages
explained: nonadditive interaction terms are substantial. Max absolute uniform
loss change is 0.000416. Predictions change in 14/128 BF16 comparisons, with
at most two changes in a 256-example batch. All occurred on the second batch.
There are 17 changed-example occurrences but only six distinct training indices
across reused comparisons; ties appear in some outputs. Treat these as repeated
descriptive checks, not independent samples or an accuracy significance test.
Archived-history key-only swaps change predictions in 2/32 BF16 comparisons.

Restoring reference key bias leaves the BF16 full-update discrepancy largely
present: median residual/full norm ratios are 0.989 (math) and 0.991 (automatic).
Across all BF16-gradient comparisons the range is 0.872–1.109. Restoration
reduces the norm in 77/128 cases and increases it in the remaining 51; there is
no reliable large reduction. It is not justified to exempt key bias, remove it,
or declare BF16 functionally equivalent from the earlier 99.86–99.94% share of
archived-history squared parameter discrepancy.

## Controls and numerical qualifications

The native-FP32-gradient control also exhibits BF16 key-only sensitivity:
max logit change 0.03125, with prediction changes in 4/32 evaluations. Restoring
reference key bias makes its BF16 evaluated logits exactly match the reference
in all 32 comparisons. Thus sensitivity is not exclusive to BF16-generated
gradients. This control and the exact repeats support deterministic finite-
precision sensitivity to parameter changes in the tested pipeline, not random
repeat noise. They do not localize every effect to a particular projection or
kernel, and the full BF16-gradient residual remains.

Changing evaluation precision at fixed reference weights gives RMS logit
differences 0.00351–0.00448 and max differences up to 0.0529, with at most one
prediction change per batch. Keep this separate from within-policy component
effects. Automatic BF16 dispatched efficient SDPA for every recorded forward;
it did not dispatch flash. Both math policies dispatched math SDPA throughout.

Native execution loss differs from uniform FP64 rescoring by -0.000143 to
+0.000185 for BF16/math and -0.000131 to +0.000166 for BF16/automatic; FP32 gaps
are below 9e-8 in magnitude. The component conclusions above use uniform saved-
logit scoring. Original gradients retain their original loss arithmetic, so
this test cannot attribute all gradient differences to attention projection.

## Decision and next step

Close this bounded diagnostic. Retain the precision-dependent functional finding
and the failed key-restoration remedy as evidence for the paper discussion.
Do not retroactively pass the earlier BF16 parameter-distance gate or infer a
speed gain. The scoped 6.34–6.40% FP32 unbind result and its failed 10% threshold
remain unchanged; established reference comparisons still constrain novelty.

Recommended next: a separately specified, short matched continuation study
comparing the qualified FP32 reference with conventional BF16/math and BF16/
automatic execution. Use the same explicit FP32 loss computation in all arms,
full trainable parameters, matched source/optimizer/data schedules and fixed
training budgets. Predeclare validation-quality and complete adaptation-time
criteria before execution. Assess whether useful end-to-end savings actually
preserve quality; do not pursue a key-bias exemption on this evidence. This is
a new prospective practical question, not a rerun or waiver of the old gate.
No next experiment is prepared or launched, and no new run command is issued.
Publication remains on hold. Two checkpoints and two fixed batches cannot
establish generality, long-run safety, statistical equivalence or novelty.

## Files and preservation

Raw evidence: results/key_bias_function_v1 (unchanged).
Detailed summaries: comparison_rows.json, group_summary.json, findings.json.
Checks: independent_audit.json, loss_arithmetic_and_dispatch.json,
prediction_changes.json, source_metadata.json.
Portable summary scripts and reviewed results:
paper_support/key_bias_function_results_v1. Frozen source capsule unchanged.
Current master v7.43; paper memory v32. Historical documents remain intact.
