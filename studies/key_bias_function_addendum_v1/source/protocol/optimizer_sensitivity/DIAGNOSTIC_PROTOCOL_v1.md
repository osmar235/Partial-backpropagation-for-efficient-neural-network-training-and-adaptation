# Optimizer sensitivity diagnostic v1 — 2026-10-27

Purpose: explain the coexistence of small aggregate gradient differences and
larger worst-coordinate AdamW parameter differences. Preserve all prior BF16
gate failures. No accuracy verdict, benchmark, new-method or superiority claim.

## Fixed scope

Six full-size trajectories, batch256: initialized seeds99201/99202 on their
original synthetic inputs; archived trained seeds1501/1502 each on synthetic
inputs and a shared fixed batch drawn only from the established45000-example
CIFAR10 training split. No validation/test examples, augmentation, MixUp or
quality evaluation. Real inputs use existing CIFAR normalization. Initialized
versus trained synthetic inputs are seed-specific, so this is descriptive,
not a factorial estimate of the causal effect of pretraining alone.

At anchors0 and2 on a two-update FP32/reference trajectory starting with fresh
AdamW, compute all six previously defined execution arms' gradients at exactly
the same parameter tensors. Disable dropout throughout, including reference
and trajectory advances. Reuse the prior strict math / permissive automatic
dispatch policies and unchanged attention wrappers. Profile actual dispatch.
Repeat the FP32/reference gradient once at each anchor and require exact
gradient/logit/loss equality. No candidate advances its own trajectory.

Replay each gradient once through actual PyTorch AdamW at identical weights
and identical optimizer state. At every anchor include a fresh-state replay.
At anchor2 also include the same two-update reference moments and counters.
For trained anchor0 additionally replay original archived moments/counters
(age17600). The latter uses the diagnostic learning rate1e-4 for both arms;
it is a controlled history transplant, not an exact continuation of the
source training schedule. Betas0.9/0.999, epsilon1e-8, weight decay0.05 and
all other optimizer options are fixed and matched. No epsilon or rate tuning.
Source parameter order, state shapes, counters, checkpoint hashes and original
optimizer group metadata must be checked. Never overwrite source checkpoints.

Full scope:12 anchors,72 distinct arm gradient captures,12 repeated reference
captures,6 intermediate trajectory gradient captures =90 forward/backward
calls.132 one-step diagnostic replays and12 reference trajectory updates.
Replays consume saved gradients and perform no additional forward/backward.
No extra seeds, epochs, timing sessions or automatic follow-up. CPU smoke skips
BF16; tiny fake trained fixtures test state loading but are not trained evidence.

## Recorded measurements

Persist raw FP32 gradients, shared anchor parameters, shared optimizer states,
actual FP32 parameter deltas and resulting optimizer state hashes. This supports
read-only recalculation of summaries without repeating forward/backward. Expect
a few GB of full diagnostic evidence; exclude raw tensors from public packages.
Store input tensors and indices/provenance. Every replay must verify the same
starting parameter and optimizer hashes, parameter coverage, finite values and
expected per-parameter step counter. Reference self-comparisons must be zero.

Report aggregate and per-tensor gradient/update relative L2, cosine and maximum
absolute difference; strict sign flips versus zero/nonzero disagreements; and
counts and squared update-difference mass by fixed reference-gradient magnitude
bins [0,1e-10), [1e-10,1e-8), [1e-8,1e-6), [1e-6,1e-4), [1e-4,1e-2), [1e-2,inf).
Zeros are retained. Report counts as well as proportions; tiny denominators
must not be interpreted as large substantive effects without absolute values.

For the32 largest absolute update differences, save coordinate names/indices,
initial parameter, reference/candidate gradients, initial moments/counter,
actual updates and FP64 analytic AdamW predictions. These are diagnostic
approximations to FP32 arithmetic, not an alternate optimizer implementation.
The auditor independently recalculates full summaries from stored tensors.
There is no new BF16 eligibility gate: finite captures are analyzed even when
they would fail historical closeness tests. Unsupported or nonfinite arms are
recorded explicitly; unexpected implementation errors abort.

Fresh versus retained history contrasts at a given anchor use the exact same
weights and gradients; contrasts between anchors also change weights. The
historical3e-4 limit applied to three diverging updates, so it is contextual
information only here, not a threshold for these matched one-step replays.
Concentration in small-gradient bins or reduction with retained moments may
support an optimizer-sensitivity explanation, but does not establish harmless
task behavior. Sign changes must be observed, not inferred from aggregate cosine.

## Integrity and interpretation

Freeze all sources, specification, environment, dataset training files and source
checkpoints before the full run. Each trajectory has a completion marker and
immutable attempts. Completed work is hash-verified and skipped. Interruptions
remain visible. An audit PASS means integrity and reconstruction passed, not
BF16 quality equivalence. No wall-time performance claim from profiled runs.

If this motivates a practical mixed-precision experiment, declare a separate
real-data quality margin and complete-workflow cost design before collecting
new quality results. All earlier thresholds and negative results remain intact.

AdamW equation reference: https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html
The installed PyTorch2.11.0+cu128 optimizer is executed; the documentation is
context for moment normalization, not a guarantee of bitwise implementation.
