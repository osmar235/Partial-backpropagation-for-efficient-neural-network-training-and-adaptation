# Research reset: useful contribution before publication

2026-10-20. Author-directed return to method development. Publication
preparation is on hold. This supersedes earlier next steps to package v7
or stop all future experimentation; completed Stage6--8 studies remain intact.
This is a research design, not a frozen claim of a new method's efficacy.

## Why the current contribution is insufficient for this objective

The original controller saves work on a ResNet recipe, but has not shown
an advantage over well-matched blind scheduling. Its ViT/MixUp transfer is
substantially worse than periodic at nearly identical time and partial use.
Moment continuation has not demonstrated a unique advantage over replay.
The author wants a practically useful method; preparing a limitation paper
does not fulfill that objective. Do not keep the current gate or continuation
rule merely to preserve the original story.

## Completed now: a diagnostic, not another long training screen

`scripts/stage9_entropy_diagnostic.py` reconstructs target entropy from exact
batch/permutation streams on the two existing DEVELOPMENT seeds1001/1002.
It uses only CIFAR training-label metadata and saved logs. No new training
or test-dataset construction. Four trajectories/211,200 updates reconstructed;
the recorded loss gate is reproduced. Formula/gradient checks passed.

For smoothed mixed targets q and model probabilities p:

    cross_entropy(q,p) = H(q) + KL(q || p)

The raw gate is exposed to changing label entropy. Weak mixing is defined
diagnostically as min(lambda,1-lambda)<=0.05; strong mixing as >=0.20.
On the two observed LGPB development trajectories, actual skip probability
was59.74--59.87% for weak mixing and0.063--0.106% for strong mixing.
The within-epoch-centered correlation of target entropy and raw loss was
0.960/0.960. This is association, not proof of the source of the accuracy loss.

Subtracting exact target entropy leaves the training gradient unchanged, but
its OFFLINE SHADOW gate still skips48.57--49.21% of weak batches versus
4.46--4.72% of strong batches. Thus entropy subtraction does not remove the
augmentation-dependent selection. It is not a sufficient redesign on the
available evidence. Shadow trajectories are fixed: these numbers cannot
predict the accuracy or wall time of a newly trained controller.

Outputs: `results/stage9A_entropy_diagnostic_v1/REPORT.md`, `diagnostic.json`,
`shadow_policy_comparison.csv`, `policy_groups.csv`, `correlations.csv`.

## Proposed direction: decide after fresh suffix backward

The question changes from "is this batch loss low?" to "is computing this
prefix's gradient worth its cost on this update?" A candidate decision point
is the prefix/suffix boundary after suffix backward. Its current activation
gradient is closer to the omitted computation than scalar batch loss, but
is not automatically a valid importance proxy or a novelty claim.

1. Preserve the ordinary forward through prefix and suffix, with a detached
   differentiable bridge at the boundary.
2. Compute exact suffix gradients and the current boundary activation gradient.
3. Use a cheap candidate signal to decide whether to continue backward through
   the saved prefix graph; on refresh, do so using the already computed
   boundary gradient. No second model forward is required by this plumbing.
4. Explicitly evaluate state handling rather than assume zero-gradient Adam
   continuation is indispensable. Do not combine gate and optimizer redesigns
   without controls separating them.

`scripts/stage9_split_backward_proof.py` PASSED on an artificial-data native
six-block ViT in train mode with matched dropout: exact logits, exact suffix
gradients before prefix backward, no prefix gradients until requested, and
bitwise-exact full gradients after continuing prefix backward. A nonzero head
avoids a vacuous zero-gradient test. This is a CPU correctness proof ONLY.
No new policy, GPU speedup, useful importance predictor or novelty is established.

## Next diagnostic, before a performance study

Prepare one bounded training-only instrumented experiment. At shared model/
optimizer states, measure candidate features available BEFORE prefix backward
(raw loss, entropy-corrected loss, boundary gradient norm and simple
activation-weighted boundary statistics). Then compute the prefix gradient
as a diagnostic label. Test prediction of prefix update error and, on a
separate validation minibatch, the immediate full-versus-omitted update effect.
Small gradient norm alone is not proof of negligible learning value.

Use early and intermediate states, not only final checkpoints; collect these
on new DEVELOPMENT seeds1201/1202 with the existing training-derived split.
No reserved evaluation is planned yet. Calibration and scoring batches must
be separated. Diagnostic full-prefix/oracle evaluations cannot be counted as
free deployment features. Account for split-backward overhead, bridge-gradient
materialization, feature reduction/synchronization, optimizer and logging cost.
Do not use a complete future window to make an allegedly online decision.

Proceed to a bounded pilot only if a deployable signal outperforms a blind
allocation in held-out diagnostic batches and measured overhead leaves
positive net GPU-time headroom. Freeze candidate, schedule/boundary and state
handling before pilot outcomes; compare full training, periodic, matched
random and candidate. Match realized resources and report time-to-quality,
not just fewer backward calls or better fixed-step accuracy.

For later efficacy screening, require a practical advantage over BOTH full
training and the best simple blind policy at comparable quality, then verify
with fresh run seeds and another workload. A possible operational screen is
>=5% measured time-to-quality improvement with final accuracy within0.2pp;
targets, margins and uncertainty must be finalized prospectively for the
actual workload. These are proposed design values, not a registered test.
If the signal or cost advantage fails, reject this direction rather than
add more normalization/EMA knobs. Do not commission another24-run sweep yet.

## Novelty and comparator checks remain required

- [LAG](https://arxiv.org/abs/1805.09965) already adapts gradient refresh/reuse
  for distributed learning. Gradient-triggered refreshing itself is not new;
  its distributed setting does not establish our single-GPU performance.
- [LayerDropBack](https://arxiv.org/abs/2412.18027) studies backward-only
  layer omission with the forward computation retained. That property alone
  is not our novelty claim.
- [Ibrahim et al., 2025](https://openaccess.cms-conferences.org/publications/book/978-1-964867-37-3/article/978-1-964867-37-3_7)
  evaluate workload skipping, describe GSkip's layer-gradient-change policy,
  and emphasize random controls, wall time and time-to-accuracy/Pareto tests.
  Their findings reinforce the need to beat simple alternatives. This read
  of the primary abstract is an initial map, not a completed novelty review.

Before claiming a new method, compare full algorithms, available information,
state transitions, overhead and applicable setting against these and other
close work. No absence of prior art or originality guarantee is asserted.

## Preservation and current status

Original results, freeze, v7 Word/PDF and release candidates are retained.
V7 is an accurate research record, not the current publication objective.
The two new CPU diagnostics are separate from previous frozen sources and
are not evidence that a replacement method works. Current master:v6.3.
Next work is the bounded mechanism/utility diagnostic, not submission work.
