# Key-bias functional-impact diagnostic v1 — 2026-10-27

The previous exploratory review localized most archived-history squared update
discrepancy to key bias. This new prospective experiment measures the functional
effect. All historical gates, failures and parameter metrics remain unchanged.

Use only the two trained models' training-input cases from optimizer_sensitivity_v1.
At anchor0 include fresh/archived states; at anchor2 include fresh/two-step retained
states. Eight source conditions. Use all four BF16 gradient origins plus native
FP32 as a numerical control, giving40 component-swap cases. No new gradient is
computed. Recover actual post-update weights by replaying each saved gradient
through the frozen AdamW recipe; require exact saved parameter-delta and final
optimizer-hash matches. Do not reconstruct weights by adding rounded deltas.
This replays existing evidence, not a new learning trajectory.

## Component controls

Let R be the recovered FP32/reference updated weights, C the recovered candidate
weights, and W the shared pre-update anchor. Evaluate W and R as common controls,
then C, R with only C's key-bias slices, and C with R's key-bias slices restored.
In every packed in_proj_bias, select exactly the middle third (K in Q/K/V order).
Copy these FP32 parameter values directly. All other coordinates must be bitwise
unchanged from their designated donor; swaps must preserve R/C source tensors.
Record full-state hashes and exact component checks. No bias is zeroed, frozen,
removed from the model, or exempted from any prior result.

Use the same pinned reference attention implementation for all forward evaluations
to isolate weight effects from implementation differences. Independently evaluate
each state under reference FP32/math, BF16/math and BF16/automatic policies.
Retain original determinism/reduction/TF32 settings and profile actual dispatch.
Keep the model in training mode with all dropout disabled, under torch.no_grad;
no backward call, parameter update during evaluation, test split or timing claim.
FP32 and BF16 weights remain stored in FP32; autocast controls arithmetic only.

## Input and scoring controls

Batch1 is the exact saved training batch that generated the diagnostic gradients.
Batch2 contains256 examples drawn from the same45000-example training split with
fixed seed394001, after excluding every batch1 index. Both models share these
batches. Check disjointness, split/data hashes, and normalization. No augmentation,
MixUp, validation or test examples. Batch2 is a disjoint diagnostic batch, not
an independent quality benchmark. Smoke fixtures are explicitly artificial.

Save logits as FP32, labels, input indices, precision/dispatch metadata and
parameter hashes. Recalculate loss uniformly in FP64 from saved logits with
label smoothing0.1. Also compute explicit FP32 loss outside autocast from those
same logits; check its agreement within 1e-5 +1e-6*|FP64 loss|. This is a
reconstruction check, not a BF16 quality margin. Retain original execution loss
separately and report its difference from uniform scoring. Artificial validation
found BF16-input autocast cross-entropy differed from explicit FP32 scoring in
the installed runtime, despite returning FP32 loss. Do not relax reconstruction
tolerance or equate these two losses. Saved gradients retain their original loss
arithmetic; this diagnostic does not attribute all discrepancies to projection.
Report within-policy logit max-absolute/RMS/L2 differences, relative L2 with an
explicit denominator floor, loss changes, prediction disagreements and correct
counts. Do not equate a training-batch correct count with generalization accuracy.

For each policy/batch compute full effect C-R, key-only effect K-R, residual
effect S-R and interaction C-K-S+R on logits and loss. Ratios of vector norms
are descriptive, undefined for zero full effect, and need not be<=1; they are
not causal percentages. Also report changes from W, and cross-policy differences
at the same R weights, separately from the within-policy component effects.
Repeat R once at the end of each policy/batch block. Require exact math-policy
repeats; report automatic-policy repeat differences as a numerical noise check.

At full availability:24 unchanged-state evaluations,48 reference evaluations,
720 candidate/component evaluations and48 repeated references =840 forwards.
There are 48 saved-gradient optimizer replays (reference plus five candidates
for each of eight conditions), and no
backward calls. Persist logits, not duplicate full models. The auditor verifies
hashes/provenance, reconstructs original optimizer replays and swaps without
forward/backward, and recomputes all summaries from stored logits. Completed
conditions are verified and skipped; interrupted attempts remain visible.

Unsupported/nonfinite policies are explicit and cannot support conclusions.
Do not relax a gate or add seeds/epochs automatically. A small observed effect
would motivate a separately specified real-data quality and workflow-cost study;
it cannot establish long-run harmlessness, speed, novelty or publication readiness.
