# Comparing research objectives — 2026-10-24

## Recommendation for the next decision

Make **the cost of adapting an existing checkpoint to a predefined quality
requirement** the recommended primary objective. Retain inference cost as a
secondary outcome. This objective has the clearest useful ordinary baseline
in our evidence and a concrete competing compression tradeoff. It is a
recommendation for the author to select, not an already approved change to
the research goal, a candidate selection or authorization for another run.

If training entirely from scratch is essential to the intended contribution,
growth is the more directly relevant line to reconsider. Its current evidence
does not establish faster attainment of a quality target. Token recovery's
approximately 1.1% source-inclusive saving remains a valid observed tradeoff;
its magnitude alone is not a reason to discard it.

All four ideas received initial tests. This comparison completes step 1 of the
agreed sequence. Step 2 would select one candidate and one unresolved hypothesis
under the chosen objective. No new experiment or confirmation study is prepared.

## 1. Training from scratch

The desired claim would be lower total compute/time to achieve a predefined
quality requirement, starting without the trained task checkpoint. A shorter
fixed run at lower accuracy does not by itself establish that claim.

| Strategy and study | Observed saving, two seeds | Quality deficit against its matched full control | What the evidence supports |
|---|---|---|---|
| Midpoint growth, Stage11B | 23.54% / 21.99% complete fixed-run time | 1.40 / 0.12 pp; original limit 0.20 pp | Substantial budget reduction, inconsistent final quality |
| Train full source, then similarity recovery, Stage14B | 1.10% / 1.07% source training + charged recovery | 0.80 / 0.26 pp; original limit 0.30 pp | Small, consistently signed cost tradeoff, inconsistent final quality |

These clocks and budgets differ. Growth used 100 epochs and a complete-run
clock; the token pipeline uses a recorded 100-epoch full source plus five
recovery epochs and sums scoped costs from separate executions. The respective
baselines match their own study. Their percentages are not a direct ranking
of two algorithms at the same final accuracy or wall-time budget.

Growth also reached the predeclared sustained 80% target **39.75% / 5.76%
later** than full training, on its secondary observed clock. Thus even its
larger fixed-budget saving is not evidence of faster learning to that target.
The post-expansion deterioration remains a concrete unresolved mechanism.

**Assessment:** no candidate has demonstrated a consistent near-full-quality
from-scratch efficiency win. Growth has larger observed fixed-budget savings
and direct relevance to this goal, but needs a causal recovery hypothesis
before any expansion of training. Keep the 1.1% token result visible with its
quality and cost-scope qualifications.

## 2. Adapting an existing checkpoint

Here the checkpoint is available before the decision to adapt it. Its common
historical training cost can be excluded from the marginal adaptation question,
while still reported separately. Include all new setup and adaptation work.
The quality requirement and timing boundary must be specified before a new run.

| Strategy | Observed time saving versus matched full | Final deficit, two seeds | Interpretation |
|---|---|---|---|
| Head warmup, then full (Stage12A control) | 11.66% / 11.38% complete adaptation time | 0.26 / 0.20 pp | Within the study's 0.30-point quality margin in both seeds; clearest useful ordinary baseline |
| Periodic prefix adaptation (Stage12A candidate) | 39.91% / 38.81% complete adaptation time | 1.52 / 0.96 pp | Fails final quality in both seeds |
| Similarity merging (Stage14B candidate) | 22.55% / 21.51% charged recovery time | 0.80 / 0.26 pp | Both resource routes pass; quality passes in one seed |
| Detached local losses (Stage13C candidate) | **8.28% / 5.69% slower**, calibration included | 8.26 / 12.18 pp | No useful efficiency case for this fixed recipe |

The head-warmup/full arm was a prespecified ordinary control. Its useful
observed result does not retroactively make it the successful primary
candidate, establish a new method or provide independent confirmation.
Periodic updates also lose 1.26 / 0.76 points to that stronger inexpensive
control, despite saving 31.98% / 30.95% of its complete time.

For the already declared secondary 92% target, periodic adaptation reached it
30.14% / 29.05% sooner than head-warmup/full on the observed clock. Preserve
that lower-target tradeoff, but do not choose 92% after seeing these data to
replace the original primary objective. A future quality requirement should
come from the intended application or a prospectively justified design.

Stage12A uses externally pretrained ResNet-18, 96px inputs, fixed BatchNorm
and twelve adaptation epochs. Stage14B uses a CIFAR-trained ViT, 32px inputs
and five recovery epochs. Stage13C uses an intermediate pretrained ResNet
state and a one-epoch trajectory. Their savings are relative to their own
controls, with different clocks. We cannot conclude that token merging beats
head-warmup/full from 22% versus 11%, or compare their absolute accuracies as
method rankings. A direct comparison needs a common workload and timing rule.

**Assessment:** strongest setting for the next objective. There is a useful
ordinary baseline to beat and measurable compression savings to investigate.
Candidate choice and novelty remain separate questions. Neither a familiar
baseline nor established merging machinery becomes novel through this comparison.

## 3. Cheaper inference

The desired claim would be lower deployment latency, resource use or energy
at an acceptable quality level. The relevant measurements must match the
deployment batch size, hardware and data path; training savings do not answer
this question by themselves.

Stage14B similarity models show **23.34% / 21.92% lower inference-operation
time**, with **15.38% lower incremental inference allocation**, versus equally
adapted full models. Final accuracy deficits remain 0.80 / 0.26 points. The
26.39% incremental memory saving previously reported is for **training**, not
inference; absolute training peak allocation falls 18.48%.

Spatial merging saves 22.53% / 23.75% inference-operation time but loses
1.30 / 0.98 accuracy points. Similarity preserves more quality at comparable
cost in this measured setting; it still misses the original quality rule.
Measurements used twelve fixed training batches of 256, with no optimizer
updates. They do not establish single-example latency, complete service
throughput, energy efficiency or deployment generalization.

Growth finishes at full width, and the adaptation/local procedures retain their
backbone inference computation. No inference advantage for those procedures
has been established in these experiments. A deployment-first project would
therefore focus on the token line, but needs its own quality/cost design and
prior-method comparisons. This would be an explicit change in emphasis from
the original training-efficiency question.

**Assessment:** plausible secondary outcome for an adaptation project, or a
distinct primary objective if deployment is the real use case. No confirmed
matched-quality inference winner yet.

## Why 1.1% and 22% can both be correct

Let P be the common source-training cost, F full adaptation cost, and C
compressed adaptation cost. Marginal saving is (F-C)/F; source-inclusive
saving is (F-C)/(P+F). The latter equals marginal saving multiplied by the
share F/(P+F).

In these runs that adaptation share is about 5%, so the same 5.93 / 5.72
seconds saved becomes either 22.55% / 21.51% marginal saving or 1.095% /
1.074% source-inclusive saving. This is a change of denominator, not another
independent replication. The numerical consistency across two seeds is
encouraging; repeated timings and broader training replication are still
needed to characterize robustness. A small reliable saving can matter, but
the acceptable accuracy tradeoff must be explicit.

## Decision boundary and retained evidence

Recommended objective: **reduce complete marginal adaptation cost while
meeting a prospectively specified quality requirement against strong ordinary
controls; report inference cost secondarily**. Use that objective to choose
the next candidate/hypothesis in step 2. Do not rank the ideas by percentage
alone, silently change original margins, or present exploration as confirmation.

No new training, model rescoring, test evaluation, protocol freeze or automatic
follow-up occurred for this synthesis. All prior integrity checks and failed
scientific decisions remain intact. The comparison recomputes numerical
contrasts from the audited tables and preserves the previous file hashes.
Two seeds per study, reused development data, different architectures and
strict FP32 execution limit the claims. Publication remains on hold.

Evidence: [growth review](../stage11B_review_v1/REPORT.md),
[adaptation review](../stage12A_review_v1/REPORT.md),
[local/token diagnostic review](../stage13C_stage14A_review_v1/REPORT.md),
[token recovery review](../stage14B_review_v1/REPORT.md).
`verified_comparison.json` contains the recomputed per-seed records;
`comparison_verification.json` records provenance checks and scope.
