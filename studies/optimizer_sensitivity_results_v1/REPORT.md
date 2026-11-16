# Optimizer-sensitivity review — 2026-10-27

Completed diagnostic audited successfully. No model forward/backward or
optimizer step was rerun during review. The saved tensor evidence supports
optimizer sensitivity as a mechanism for gradient-to-update amplification.
Retained history is not uniformly protective. A post hoc partition localizes
most of the archived-history discrepancy to attention key-bias coordinates.
Neither task-quality preservation nor a new speed benefit is established.

## Integrity and scope

Six unfragmented trajectories, 12 anchors, 72 finite distinct arm captures,
132 replay comparisons (88 BF16, 22 native FP32 controls, 22 reference self
comparisons). Reference repeats are exact and self-comparisons zero. All frozen
source/specification/checkpoint/data hashes and completion hashes pass. The
frozen auditor reconstructed stored summaries; a separate CPU implementation
recomputed aggregate norms, signs, bins and coordinate AdamW predictions from
the raw tensors. Both checks pass. Raw anchors occupy about 2.55 GB.

These comparisons reuse two initialized seeds, two trained checkpoints, fixed
synthetic inputs and one shared training-only batch. Anchors follow two short
FP32 updates with dropout disabled. No test/validation accuracy, training curve,
profiling speed estimate or independent population inference is available.
The archived states use the common diagnostic rate 1e-4, not their original
training-schedule rate. Cross-anchor/model/input comparisons are descriptive;
within-anchor history contrasts share exactly the same weights and gradients.

## Observed mechanisms

Fresh AdamW magnifies some small coordinate sign disagreements: across all 48
fresh BF16 comparisons, every one of the saved 32 largest update discrepancies
involves opposite gradient signs (1536/1536 selected coordinates). The maximum
parameter-step differences are approximately 2e-4, twice the 1e-4 learning rate.
This is selected-coordinate evidence, not a statement that every coordinate
flips. Overall sign-flip fractions span about 0.12–0.73% across the diagnostic.
For fresh states, more than 99.84% of squared update discrepancy lies in the
predeclared bins with reference |gradient|<1e-4. This numerical magnitude is
scale-dependent; no universal small-gradient threshold is claimed.

Retaining the two recent reference steps reduces relative update L2 discrepancy
in 24/24 matched BF16 comparisons. Relative discrepancies fall by about 75.0–88.6%
versus resetting history at those same weights. On the real training batch,
fresh update relative L2 is 13.51–16.09%; two-step history gives 1.66–1.91%.
This is a reduction of disagreement, not an accuracy gain or timing saving.

Archived 17600-step history behaves differently. For the real training batch,
it increases relative update disagreement in 8/8 matched comparisons: fresh
12.37–14.04% versus archived 14.40–20.86% (paired increases 13.69–62.04%). On
synthetic inputs, archived history lowers relative L2 but raises maximum
coordinate disagreement. Thus 'keep optimizer history' is not a general fix.

Absolute-versus-relative qualification: on the training batch, archived history
reduces the absolute L2 norm of the update discrepancy by 66.69–74.16% in all 8
comparisons, even though relative L2 and the worst-coordinate difference rise.
The reference update norm shrinks more strongly, changing the relative-error
denominator. These metrics answer different questions; the result does not
mean archived history is uniformly worse. norm_scale_check.json records all
matched absolute/relative norm contrasts. No metric replaces the historical gate.


The largest archived differences are about 2.86e-4–3.13e-4; these one-step
values must not be judged as a retest of the historical three-update gate.

| Model / input | Anchor | History | Gradient relative L2 | Update relative L2 | Key-bias share of squared discrepancy |
|---|---:|---|---:|---:|---:|
| initialized / synthetic | 0 | fresh | 1.07–1.13% | 11.92–12.24% | 1.98–2.43% |
| initialized / synthetic | 2 | fresh | 1.33–2.16% | 13.07–14.61% | 1.35–2.06% |
| initialized / synthetic | 2 | retained_two_updates | 1.33–2.16% | 1.75–2.07% | 35.86–56.15% |
| trained / synthetic | 0 | archived | 0.28–0.39% | 3.85–7.23% | 98.97–99.68% |
| trained / synthetic | 0 | fresh | 0.28–0.39% | 6.20–7.70% | 6.35–9.66% |
| trained / synthetic | 2 | fresh | 0.33–0.47% | 7.02–8.33% | 5.30–7.68% |
| trained / synthetic | 2 | retained_two_updates | 0.33–0.47% | 1.47–1.78% | 91.70–96.33% |
| trained / training | 0 | archived | 1.22–1.79% | 14.40–20.86% | 99.86–99.94% |
| trained / training | 0 | fresh | 1.22–1.79% | 12.37–14.04% | 2.01–2.44% |
| trained / training | 2 | fresh | 1.84–2.88% | 13.51–16.09% | 1.36–1.98% |
| trained / training | 2 | retained_two_updates | 1.84–2.88% | 1.66–1.91% | 56.31–72.63% |

Ranges span two seeds/checkpoints and four BF16 execution arms per row. These
arms are related repeated measurements, not eight independent training runs.
The FP32 native controls are much closer: maximum one-step disagreement across
all 22 controls is 6.204e-6. The analytic AdamW predictions at the saved top
coordinates agree with actual updates to within 4.375e-8, consistent with the
small rounding differences expected between FP64 algebra and FP32 execution.

## Exploratory key-bias localization

The source orders packed bias as Q/K/V, each 192 coordinates. The K slice is
indices 192:384 in each of six layers: 1152 of 2693578 model coordinates (0.0428%).
This partition was introduced during review after observing packed-bias traces;
it is explicitly post hoc. All original coordinates remain in every full-model
metric and gate. No bias was removed, frozen or exempted.

For archived optimizer history, the largest coordinate discrepancy belongs
to K bias in all 16 BF16 comparisons. K bias accounts for 98.97–99.68% of squared
update discrepancy on synthetic inputs and 99.86–99.94% on the training batch.
The maximum reference key-bias gradient is roughly 5.24e-10–1.40e-9 across these
cases, versus 3.01e-6–1.57e-5 for BF16. Other coordinates remain measured; key
bias is not the main source of fresh-state discrepancy (about 1.35–9.66% there).
It therefore cannot alone explain all earlier failures or justify exempting
this parameter group from the historical protocol.

For this attention implementation, qk_norm=False and attention weights are
softmax(QK^T/sqrt(d)). In exact arithmetic, adding a common key bias adds the
same scalar to every key logit within a query row, which cancels in softmax.
This mathematical inference follows from the
[documented attention equation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and the pinned local Q/K/V ordering. In BF16, projection rounding can break
that exact cancellation. We have not tested the functional effect of the
observed key-bias drift, so parameter redundancy in exact arithmetic is not
proof of harmless mixed-precision behavior.

The coordinate traces also show why optimizer moments matter: small stored
second moments can give changed gradients substantial normalized updates.
This interpretation is consistent with the
[AdamW update equation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
and directly checked coordinate predictions; it is not a new optimization
algorithm. The original three-update screen still failed as declared.

## Recommended next step

Run a bounded functional-impact diagnostic using the saved trained weights and
observed updates. Compare the full reference and BF16 updates, a reference update
with only the BF16 key-bias component substituted, and a BF16 update with the
reference key-bias component restored. Evaluate loss/logit changes under matched
FP32 and BF16 policies on the original and a second fixed training-only batch.
Specify this design prospectively before execution. This isolates the functional
contribution of the identified coordinates without changing the old gate or
assuming that smaller parameter differences mean better predictions.

No such forward computation or new experiment was run/prepared in this review.
If the functional evidence supports proceeding, the following decision would
still require a separately frozen real-data quality/noninferiority and complete
workflow cost study. No present result establishes BF16 accuracy equivalence,
speedup, publication readiness or novelty. Preserve the original 6.34–6.40%
scoped unbind finding, the failed 10% gate, and established-reference controls.
