# LGPB-MC Reviewer Follow-up — Stage 7 (local prospective plan v1.1)

**Date:** 2026-10-14  
**Owner:** Osmar Pinto Neto  
**Scope:** New post-confirmation robustness and mechanism tests motivated by an AI reviewer report, **not** a revision to the frozen Stage 6B confirmatory analysis.

## Research integrity boundary

- Original Stage 6B data: CIFAR-10 ResNet-18-GroupNorm, nine held-out seeds **41–49**. The original code, results, preregistration, and confidence intervals must remain **unchanged**.
- Stage 7 experiments are **new**, prospectively enumerated here, and are *not* part of the original Stage 6B pre-specified statistical gates.
- Never relabel Stage 7 results as originally preregistered evidence. Report any failure as well as any success.
- Never update the frozen LGPB controller parameters (rho=0.95, prefix stem+layer1, forced full after a partial, explicit zeros on prefix, exact suffix gradients) in the primary LGPB variants. Other listed variants deliberately change one component for ablation.
- All substantial GPU training takes place **on the user's local RTX 5080**. Local validation includes mechanics tests, frozen-trajectory parity, and tiny artificial-data CPU and RTX 5080 ResNet checks. No full Stage 7 CIFAR experiment has run.
- **Do not select algorithms, hyperparameters, or seeds after examining new data.** Separate programmatic checks of reproducibility are allowed; describe any necessary bugfixes and rerun affected variants.

## Data, model and training constants

- Stage 7B and 7C: CIFAR-10 50,000 train / 10,000 test, original ResNet-18-GN architecture, batch size 256 (including final short batch), random 4-pixel padded crop and horizontal flip, same channel normalization.
- Stage 7D: CIFAR-100 50,000/10,000, ResNet-18-GN with a 100-class classifier, CIFAR-100 channel normalization, otherwise the Stage 6 augmentation/schedule. Changes both the dataset and output head, **not** the architectural family.
- Stage 7E: CIFAR-10 AdamW with decoupled weight decay `0.0005` (optional; same base LR without an LR re-screen). This does **not** demonstrate a fully optimized modern training recipe.
- 30 epochs, 5,880 mini-batch updates; per-update cosine LR with starting rate 0.0003, final 1% floor; base optimizer Adam defaults unless Stage 7E is selected. End-to-end training time includes augmentation, forward, backward and optimizer operations; epoch validation and algorithm-specific warm-up are excluded.
- Algorithm-specific 196 warm-up updates occur on a distinct RNG stream; all models and optimizer states are reset for each timed run. The same initialization, mini-batch ordering and augmentation streams are used within a seed. Variants rotate order cyclically across the six seeds.
- Existing data selection/timing convention is preserved as closely as possible; modified instrumentation and execution order can change measured absolute time. Primary Stage 7 comparisons are **within** their own matched runs, not against old Stage 6 timings.

## Stage 7A: no-training archived audit

- Read only frozen Stage 6B `epoch_trace.csv` and `trajectory_summary.csv` from seeds 41–49.
- Output nine-seed epoch-wise partial fraction `q`, candidate loss, and aggregate augmentation/forward/backward/optimizer time. **The old records do not distinguish full-step vs partial-step backward times**; new instrumentation is provided in Stage 7B/C.
- The archived grand-mean partial fraction was **0.35990173847316703**. This was measured on old Stage 6B before choosing the new reviewer-test schedules.

## Stage 7B: critical mechanism ablations (CIFAR-10 seeds 51–56)

Run a six-variant, cyclic-order experiment with unchanged training protocol:

1. `adam`: full-gradient Adam every update.
2. `lgpb`: **the frozen controller**, same pre-update EMA condition, rho=0.95, forced-full, explicit zero prefix grads, Adam moment continuation.
3. `random_matched`: no loss gate; **within each epoch**, exactly `round(q_epoch*196)` nonconsecutive random partial steps, using archived original Stage 6B nine-seed mean q(epoch) (uniform over feasible masks; fixed RNG seed); **moment continuation retained**.
4. `periodic_matched`: no loss gate; **within each epoch**, the same count of nonconsecutive, approximately evenly spaced partial steps; **moment continuation retained**.
5. `freeze_prefix`: same original loss gate but `param.grad=None` for prefix parameters on partial steps, so **neither the prefix weights nor their Adam moments/step counter advance**.
6. `replay_prefix`: same original loss gate but skip the prefix Adam state step on partial updates and repeat the displacement from the preceding full step. True copy/replay overhead is timed, so wall-time comparisons must account for that additional work.

**Important**: random/periodic controls are matched to the **original archived 30-epoch q(t) profile**, whose overall mean was 35.990%. Specifically they select `round(q_epoch*196)` partial steps in each epoch, for a total of 2,116 of 5,880 updates (35.986%). They do **not** match each new seed's endogenous loss-gated decisions; that would use information from the new experiment. Moment-freeze/replay variants use the same gate formula, but may induce different decisions later because the optimization trajectory changes. Record and disclose realized q; a strictly fixed-decision trace would be a future additional control.

Predeclared descriptive comparisons: `lgpb` vs `random_matched` and `periodic_matched` for the value of loss-informed decisions; `lgpb` vs `freeze_prefix` and `replay_prefix` for the value of moment continuation; all six compared with `adam` for quality and compute. Report per-seed paired final accuracy and total wall time, plus q, update-mode backward timing and optional time-to-quality. Do not apply Stage 6B original inferential gates to these post-confirmation analyses.

## Stage 7C: sensitivity study (CIFAR-10 seeds 57–62)

Six variants in matched cyclic execution order:

- `lgpb`: frozen base, rho 0.95, forced-full, stem+layer1 prefix.
- `rho_090` / `rho_099`: EMA decay 0.90 / 0.99, respectively; everything else unchanged.
- `no_forced_full`: same gate but consecutive partials allowed (riskier ablation); record max consecutive skip streak and any instability.
- `prefix_stem`: smaller prefix; only conv1 and first GroupNorm get omitted gradients.
- `prefix_stem_layer12`: larger prefix; stem + layer1 + layer2 get omitted gradients.

Log q, accuracy, wall time, and empirical mean backward duration in each mode. A different partition changes both *fraction of parameters* and backward arithmetic; do not assume one predicts the other.

## Stage 7D: CIFAR-100 transfer (new seeds 63–68)

`adam` and `lgpb`, same 30-epoch LR schedule, EMA 0.95 and forced-full rule. CIFAR-100 creates a new 100-class classifier. This is evidence of **dataset transfer with the same ResNet architecture family**, not transfer to a new architecture or a fully tuned CIFAR-100 recipe. Report both absolute accuracy and paired efficiency/accuracy deltas. If the common LR yields poor absolute performance, discuss that limitation rather than post-hoc adjusting the reported comparison.

## Stage 7E: optional AdamW robustness (CIFAR-10 seeds 69–74)

`adam` and `lgpb` both using AdamW, 5e-4 decoupled weight decay, same starting LR 3e-4. The label `adam` in these runs means the **full-gradient AdamW baseline**, recorded in `summary.json` as `optimizer=adamw`. This is a new recipe robustness check, not a new optimization of each variant.

## Timing and instrumentation caveats

- `epoch_trace.csv` includes per-epoch partial/full counts and *mean operation-based step times* for each type. `summary.json` splits backward time by full and partial steps and includes actual end-to-end training wall.
- Report differences between `training_wall_seconds` and the sum of component timers as timing/unattributed overhead, not as additional GPU compute savings.
- Timing audits synchronously sample GPU activity; results are equipment- and implementation-specific.
- In replay, cloning parameter vectors is part of the algorithm and is timed. Other methods do not pay this extra cost.
- A run is complete only when its `summary.json` exists. If interruption occurs, the launcher resumes incomplete/missing variants. Rerunning completed variants requires `--force`.

## Pre-submission scientific/reporting concerns

1. The supplied AI review says "Frontiers in Neuroscience, submitted October 14" while the supplied manuscript says "Submitted to Neural Networks, October 13". **Verify journal and actual submission status**. Do not leave a false submission statement on the manuscript title page.
2. The method actually implements **Adam moment continuation**, not an Adan-specific variant. "Two confirmatory optimizer-family comparisons" means two baseline families under the same data/timing protocol, **not transfer of LGPB-MC to Adan**.
3. The original Adan learning-rate screen used development-seed performance on CIFAR-10's evaluation/test set, so seeds 41–49 provide independent run seeds rather than a completely untouched *test dataset* for model selection. Be precise about which independence is claimed.
4. Distinguish systematic gains in end-to-end wall-time from savings estimated by theoretical FLOPs. No energy savings were measured.
5. Existing benchmark with GroupNorm, no weight decay and 30 epochs is not a state-of-the-art CIFAR-10 training recipe. The original validated claim is still defensible, but should not be generalized beyond the protocol.

## Documents to share after each phase

- `results/stage7A_original_audit/archived_epoch_q_summary.csv` and `archived_component_timing_by_seed.csv`
- `results/stage7B_core_ablation/stage7_descriptive_paired_estimates.csv`, `stage7_all_runs.csv`, `stage7_summary.md` and `seed_*/*/epoch_trace.csv` / `summary.json`
- analogous Stage 7C/7D outputs

Do not add Stage 7 outputs to the already frozen Zenodo v1.0.0 release. Archive a new dated release/version after the researcher's review and approval of new results.

## Local revision v1.1 — 2026-10-14, before substantive Stage 7 runs

The supplied review and protocol were treated as proposals and evidence, not as
instructions overriding the owner's request. This section records the adopted
changes. It supersedes conflicting execution/environment statements above.
The supplied v1.0 is preserved separately for provenance.

### Execution priority and interpretation

Run **7A (completed), 7B, 7D, 7C**, with **7E optional**. This puts the essential
mechanism controls and dataset-transfer evidence ahead of parameter sensitivity.
The numbered seed blocks and all model/training settings remain as supplied;
execution priority does not permit changes based on favorable or unfavorable
results. No new non-inferiority pass/fail claims are planned for Stage 7.

Stage 7B has 36 training runs, 7D has 12, 7C has 36, and optional 7E has 12.
Review 7B for implementation failures and scientific interpretation before the
other phases. Any correction gets a dated record and a fresh output directory
for the complete affected comparison, preserving prior attempts.

Random and periodic controls match archived **epoch counts**, not the loss-gated
decisions of a new run. Their first step each epoch is full, whereas the loss
gate can select a partial at an epoch boundary. The random mask is uniform over
feasible masks subject to that first-full restriction. Disclose both details.
Freeze/replay trajectories can change their own gate decisions. These controls
test whole algorithm behavior; they do not perfectly isolate moment dynamics
at an identical decision trace. Report realized q and avoid claiming unique
mechanism identification from these controls alone. A fixed common-mask study
can be prospectively designed later if needed; it is not silently added here.

For every Stage 7B control, report accuracy and wall ratios versus Adam as well
as the mechanism contrasts versus LGPB. CSV signs are **variant minus reference**
for accuracy and **variant/reference** for time. Confidence intervals are paired,
exploratory and unadjusted across comparisons; no selective reporting of seeds,
contrasts or accuracy targets. Time-to-quality uses epoch evaluation crossings,
without interpolation. Always report unreached targets. CIFAR-100 supports
dataset transfer only; it supplies no second architecture.

### Local installation and safeguards

- Active root: `C:\Users\osmar\projects\Projeto_AI_Fronteira`.
- New code: root `scripts/stage7_*.py` and `scripts/run_stage7*.cmd`.
- Imports support this flat working tree and the supplied `src/` layout.
- Audit launcher reads existing root `stage6b_heldout_seed41` through `49`.
- Training launchers share root `data/`, so CIFAR-10 is reused. CIFAR-100 is
  downloaded through torchvision on first 7D execution if absent.
- Root `results/stage7*` contains new outputs. `github/`, the v1.0.0 ZIP,
  original source/preregistration and original result files stay unchanged.
- Each seed records code hashes, environment, settings and a fingerprint.
  Resume requires matching settings/code/environment. A mismatch fails and
  requires a new `--out`; it never silently reuses incompatible results.
- Interrupted variants restart from initialization, not a mid-epoch checkpoint.
  Atomic `summary.json` is the completion marker. Completed variants skip.
  Direct Python `--force` reruns a matching configuration only.
- Launchers stop on a failed seed and require the exact planned seed/variant
  matrix before generating the final study report. Timing summaries ignore
  incomplete variants. Nonfinite training loss fails the run.
- Training wall includes synchronization and per-step logging overhead.
  `timing_unattributed_seconds` is wall minus the four component timers; it is
  not an extra compute saving. Use comparisons within this instrumentation.

### Reporting work after the experiments

Keep the Stage 6B and Stage 5 failure evidence intact. Describe independent
seeds rather than an untouched test dataset. Describe Adan as a baseline,
not as an implemented LGPB optimizer family. The AI critique is not established
as formal journal peer review; actual journal/submission metadata remain an
author-verification item and were not inferred or edited in manuscript files.

Related-work sources checked on 2026-10-14:

- [KAKURENBO](https://arxiv.org/abs/2310.10102): loss/confidence-based example
  hiding, rather than prefix-gradient omission; NeurIPS 2023.
- [Progressive Data Dropout](https://arxiv.org/abs/2505.22342): the source title
  is *An Embarrassingly Simple Approach to Faster Training*, correcting the
  supplied title. Discuss reduced data exposure separately from partial backward.
- [Backpropagated Output Momentum](https://arxiv.org/abs/2609.36738): output-space
  optimizer history; contextual literature, not a direct matched skip control.

No numerical cross-paper speed ranking is planned. Manuscript files are not
rewritten with Stage 7 outcomes before those outcomes exist.

