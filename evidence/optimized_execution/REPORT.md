# Optimized-execution feasibility review — 2026-10-27

Completed evidence audited; no training or timing was rerun. Both the frozen
auditor and the independent standard-library reconstruction pass. The screen
closes with BF16 ineligible under the declared numerical policy, and a repeated
FP32 operation-cost advantage for the established upstream reference.

## Execution and provenance

Four distinct, unfragmented session processes; 192 successful trials, 6,336
post-restore measured updates, 192 discarded first invocations, 960 warmups,
624 dropout-free qualification updates and 96 profiled updates. No blocked
baseline, failed timed trial, replacement attempt or missing condition.
Frozen sources, protocol, compiler capability record and completion hashes
match. All earlier immutable evidence is preserved. Every saved numerical
decision was rechecked; aggregate relative L2 values and maximum errors were
reconstructed from per-tensor statistics. Original gradient tensors are not
saved, so this does not independently rerun their numerical calculation or
reconstruct gradient sign changes. Data hashes were also checked by the frozen
auditor. See independent_audit.json, numerical_checks.json and dispatch.json.

## FP32 result

Both native and reference FP32/math passed all 32 arm-condition gates under
unchanged tolerances. Within each arm, all repeated timed learning records are
exact across sessions. Cross-arm learning records are not bitwise identical;
maximum recorded 33-update loss difference is 4.768372e-7. Final 33-update
parameter distances are not measured by stored state hashes.

| Seed | Batch | Reference saving vs native FP32 | Positive session medians | Faster paired trials |
|---|---:|---:|---:|---:|
| 99201 | 200 | 6.9674% | 4/4 | 22/24 |
| 99201 | 256 | 8.6930% | 4/4 | 24/24 |
| 99202 | 200 | 5.7833% | 4/4 | 19/24 |
| 99202 | 256 | 9.4775% | 4/4 | 24/24 |

Each value is the median of four session medians, each based on six paired
ratios. Reference wins all 16 session-condition medians and 89/96 paired trials.
Allocated peak memory is about 112.1–114.6 decimal MB lower at batch200 and
153.2–154.1 MB lower at batch256. These are synthetic operation measurements,
not complete adaptation time or real-data quality. They reinforce the earlier
competitive result; they are not evidence of a new unbind method.

The six-arm planned positions remain balanced, but excluding four BF16 arms
leaves a 5:1 or 1:5 native-first/reference-first split within a session-condition.
Reversal balances relative order across four sessions (12 pairs per order per
seed/batch). order_summary.json reports both orders; all eight order-group
medians remain positive for the reference. These are descriptive repeated
measurements on one GPU and two initialized seeds, not additional training
replicates or a confirmatory statistical test. No pooling with older screens.

## BF16 result and interpretation

All four BF16 arms were supported and completed their numerical checks. Math
used math SDPA; automatic dispatch used memory-efficient SDPA in all 32 automatic
arm-condition probes, with six layers each. Flash and cuDNN attention were not
observed. Automatic policy permits nondeterministic execution; this was explicit
in the prospective design. Hardware support is distinct from gate eligibility.

All 64 BF16 arm-condition evaluations failed only the parameter-max-absolute
sanity limit: 3e-4. The limit passed after update1 and failed after updates2/3
in every evaluation (128 failed checks). Maximum observed parameter difference
was 5.756542e-4. Worst offending tensors span MLP and attention weights/biases;
this is not confined to the previously observed packed attention bias.

All other declared BF16 checks passed: global gradient relative L2 was about
1.063–2.040%, gradient cosine at least 0.99979225, maximum loss difference
0.0005288124, and both optimizer moment budgets passed. Small aggregate gradient
differences and larger worst-coordinate parameter differences coexist. We have
not established their cause, impact on predictions after longer training, or
real-data accuracy. The artificial model starts from initialized weights,
not the trained checkpoints used in the earlier adaptation study.

BF16 received no eligible performance trials. There is consequently no BF16
speedup, memory comparison or task-quality verdict to report from this screen.
It would be incorrect to call BF16 universally unsuitable, to say it harms
accuracy, or to loosen the old gate and relabel this run a success. Preserve
the failed result and the engineering nature of the original limit.

Compilation was unavailable in the recorded local probe (TritonMissing).
No compiled performance was measured and no environment change was made.

## Recommended next decision

Before another long benchmark, use a small, separately specified optimizer
sensitivity diagnostic: compare FP32 and BF16 gradients at identical parameter
and optimizer states; record per-coordinate sign differences, gradient magnitude,
actual AdamW update differences and affected tensors. Include a trained-state
control to distinguish fresh optimizer/initialization sensitivity from the
intended adaptation setting. Keep input, dropout policy and optimizer state
matched, with no epsilon/learning-rate tuning to pass this historical gate.

Hypothesis, not a demonstrated explanation: AdamW's coordinatewise moment
normalization may turn small changes in some gradients into larger differences
in individual updates. Its documented update divides the first moment by the
square root of the second moment plus epsilon. See the official
[AdamW definition](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
Lower precision and alternative operations can change arithmetic results;
[PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
provide context, not evidence that this specific result is harmless.

That diagnostic must not rescue or overwrite this screen. If it motivates a
practical BF16 study, freeze a new real-data accuracy/noninferiority criterion
and complete-workflow timing design before collecting new quality results.
Numerical closeness and quality preservation answer different questions.
No next experiment is prepared or launched here; this recommendation is ready
for the author's decision. All four original ideas already received initial
tests. Current work is an execution follow-up. Publication remains on hold.

Retain the scoped 6.34–6.40% original real-data unbind saving, exact recorded
outcomes, and failed original 10% gate. The established-reference evidence
continues to prevent a consistent unbind-superiority or novel-method claim.
