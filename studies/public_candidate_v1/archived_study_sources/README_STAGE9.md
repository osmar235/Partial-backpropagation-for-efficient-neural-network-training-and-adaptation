# Current decision: real-data QKV saving confirmed — 2026-10-26

Independent review confirms6.34% /6.40% median complete adaptation time reduction;
all16 paired comparisons faster. Paired losses,epoch logits,model/Adam/RNG states
identical; accuracies84.70% /83.88%. All32 final checkpoint predictions reproduce.
Training/augmentation alone saves6.97% /7.24%. Absolute peak memory unchanged.
Original10% screen remains failed; separate real-data confirmation passes.
Review results/qkv_training_review_v1/REPORT.md. Master v7.34; paper memory v23.
Broader generality/optimized baselines/novelty remain open; no next run prepared.
Prior/raw evidence unchanged; publication on hold. Older statuses are historical.

# Current action: QKV real-data confirmation ready — 2026-10-26

Run scripts/run_qkv_training.cmd (README_QKV_TRAINING.md).
Eight sessions, native/unbind, two existing source states, five epochs each;
32 trajectories /28,160 training updates. Primary: complete adaptation workflow
including loading/setup/augmentation/validation/checkpoints, with matched quality.
CPU/CUDA small workflows, identity, checkpoint rescoring and audits pass. Frozen
real data/source/reference fingerprints verified. Full real-data study not run.
Return results/qkv_training_v1. Master v7.33; paper memory v22.
Confirmed operation saving and old10% failure preserved. Publication on hold.
Older statuses and commands below are historical.

# Current decision: QKV saving confirmed — 2026-10-26

Completed confirmation independently audited. Median session savings 7.14%,
5.23%, 6.17%, 5.62%; all 32 session-condition medians positive; 123/128 pairs
faster. Adjusted p=.015625 per condition under the prespecified session model.
Checked numerical differences zero; paired 129-update losses/final weights/Adam/RNG
identical. Memory unchanged. Original >=10% screen remains failed.
Next: matched real-data complete adaptation comparison, not yet prepared or run.
Review results/qkv_confirmation_review_v1/REPORT.md. Master v7.32; memory v21.
Prior/raw evidence unchanged; publication on hold. Older statuses are historical.

# Current action: QKV saving confirmation ready — 2026-10-26

Run scripts/run_qkv_confirmation.cmd (README_QKV_CONFIRMATION.md).
Eight fresh sessions, new seeds, longer paired timing blocks; original candidate
unchanged. Five workflow tests plus original CPU/CUDA candidate tests and reduced
pipelines pass. Full confirmation is pending; no real training launched.
Analyze session-level reproducibility/magnitude, not a retrospectively lowered
practical threshold. Original >=10% screen stays failed. Send results/qkv_confirmation_v1.
Master v7.31; paper memory v20. Earlier evidence preserved; publication on hold.
Earlier statuses and commands below are historical.

# Current decision: QKV layout saves 5–7%; practical gate not met — 2026-10-25

Audited savings: 6.62%, 5.36%, 7.14%, 5.21%; faster in 23/24 paired rounds.
All full-shape checked numerical differences zero; paired final weights/Adam/RNG
and losses identical. Verified 18 fewer independent select-backward nodes.
Allocated peak/incremental memory unchanged. No condition meets >=10% saving.
Close fixed feasibility screen; retain modest engineering result and possible
future control. No follow-up prepared or launched. Publication stays on hold.
Review results/qkv_layout_review_v1/REPORT.md. Master v7.30; paper memory v19.
Prior/raw evidence preserved; earlier statuses and commands are historical.

# Current action: QKV unbind layout screen ready — 2026-10-25

Run scripts/run_qkv_layout.cmd (README_QKV_LAYOUT.md).
One select-to-unbind split change; same math attention, FP32, full gradients and
determinism. Five CPU/five CUDA tests and both small pipelines pass with zero
checked numerical difference. Shared split backward removes six independent
select-backward nodes in the two-block check. Full-shape cost is unmeasured.
Frozen two-seed/two-batch screen; require >=10% paired saving in every condition.
No relaxed tolerance or key-bias exception. Send results/qkv_layout_v1.
Master v7.29; paper memory v18. Earlier evidence preserved; publication on hold.
Earlier statuses and commands below are historical.

# Current decision: backend screen reviewed; no alternative qualifies — 2026-10-25

Experiment complete. Final audit error was a relative/absolute-path CLI bug.
Fixed read-only entry point: scripts/review_efficient_backprop_completed.cmd.
Already verified both path forms; raw outputs unchanged. No experiment rerun.
Efficient attention: 10.46–12.35% median paired slowdown in three passing
numerical conditions; fourth blocked by parameter gate. Flash/cuDNN unsupported
under current build/policy. Close this fixed backend-only screen; retain math.
Master v7.28; paper memory v17. Review results/efficient_backprop_review_v1/REPORT.md.
Prior evidence preserved; publication on hold. Older commands/statuses historical.

# Current action: efficient full-backpropagation backend screen — 2026-10-25

Run scripts/run_efficient_backprop.cmd (README_EFFICIENT_BACKPROP.md).
Artificial FP32/strict-determinism backend comparison; all gradients retained.
Actual-shape dispatch and numerical checks precede timing. CPU/CUDA pipeline
validation passes, but the small efficient-attention arm fails two key-bias
parameter checks; it is NOT qualified. Full-shape gates remain binding.
No tolerance relaxation, silent fallback, precision change or real training.
Full screen pending. Send results/efficient_backprop_v1 with subfolders.
Master v7.27; paper memory v16. Earlier evidence preserved; publication on hold.
Earlier statuses and commands below are historical.

# Current decision: runtime profile reviewed — 2026-10-25

Sampled execution remains slower in all four conditions (3.90–14.99%).
MLP dW is only 7.66–8.56% of attributed native device time; sampling adds 360
device operations per update. Keep this implementation parked. Recommend an
exact-gradient attention/backend/layout feasibility study; none prepared yet.
IMPORTANT: frozen phase summary misses most worker-thread backward GPU work.
Use corrected trace attribution in results/runtime_profile_review_v1/REPORT.md.
Raw outputs and frozen sources preserved; no benchmark/training rerun needed.
Master v7.26; paper memory v15. Publication on hold. Earlier statuses are historical.

# Current action: runtime diagnostic ready — 2026-10-25

Run scripts/run_runtime_profile.cmd (README_RUNTIME_PROFILE.md).
Artificial only: native, exact-custom and parked sampled-custom; 12 conditions,
336 measured updates plus 36 burn-in. Measures costs and instrumentation overhead;
no new method, real-data accuracy test or speed gate. Full diagnostic pending.
CPU/CUDA correctness and small profiling pipelines passed, including identical
instrumented updates and GPU operator attribution. Send results/runtime_profile_v1.
Master v7.25; paper memory v14. Sampled-cost failure and earlier evidence preserved;
publication on hold. Earlier statuses and commands below are historical.

# Current decision: sampled weight-gradient cost reviewed — 2026-10-25

Fixed sampled-weight implementation FAILS all four artificial cost conditions:
4.97–13.82% slower than native; zero of 24 paired rounds reaches 10% saving.
All-token custom control is slower still. Correctness/audit pass; memory savings
do not replace time. Park this implementation; do not launch real-data training.
No replacement experiment prepared or launched. Master v7.24; paper memory v13.
Review: results/sampled_weight_cost_review_v1/REPORT.md. Earlier evidence preserved;
publication on hold. All older statuses and commands below are historical.

# Current action: sampled weight-gradient feasibility — 2026-10-25

Author approved Stage15A side-recipe closure. Next: full forward/dense input
gradients, sampled MLP weight gradients; all backbone parameters still trainable.
CPU/CUDA correctness and CPU pipeline checks passed. GPU cost unmeasured.
Run scripts/run_sampled_weight_cost.cmd (README_SAMPLED_WEIGHT_COST.md).
Output results/sampled_weight_cost_v1. Artificial only; 72 trials; require >=10%
paired median update saving in all four conditions before a real quality study.
No real data/checkpoint used, no quality study launched, no novelty claim.
Master v7.23; paper memory v12. Prior evidence preserved; publication on hold.
Earlier statuses and commands below are historical.

# Current decision: Stage15A reviewed — 2026-10-25

Multi-depth side: 82.60% / 82.68%, deficits 2.12 / 1.20 points against the
strongest controls. Both seeds FAIL the 0.30-point quality rule despite complete
adaptation savings of 26.19% / 20.71%. Full references qualify and reproduce.
All source/final logits match; frozen-backbone and optimizer audits pass.
Close this fixed recipe; retain resource/quality tradeoffs and prior art.
No next experiment prepared or launched. Master v7.22; paper memory v11.
Review: results/stage15A_review_v1/REPORT.md. All prior evidence preserved.
Publication on hold. Older statuses and commands below are historical.

# Current action: Stage15A side adaptation — 2026-10-25

Run scripts/run_stage15A_side_adaptation.cmd (README_STAGE15A.md).
Ten trajectories: full, head, head-then-full, final-feature side, multi-depth side;
two original full sources, five epochs each. Output results/stage15A_side_adaptation_v1.
Require quality within 0.30 points and >=10% complete adaptation-time saving
against full and fastest qualified control in both seeds; inference secondary.
Artificial CPU/CUDA validation and read-only source mapping passed. Real study
has not run. No automatic follow-up. Master v7.21; paper memory v10 unchanged.
Prior evidence preserved; publication on hold. Earlier statuses are historical.

# Current decision: side-network operation cost — 2026-10-25

Unchanged side prototype PASSES all four predeclared operation-cost conditions.
Prepare and freeze the ten-run real quality screen next. This check does not launch real training.
Artificial operation timing only; no real quality or complete adaptation result.
Master v7.20; paper memory v10. Results: results/side_cost_v1/REPORT.md.
Decision: protocol/side_cost/POST_COST_DECISION_v1.md.
Prior evidence preserved; publication on hold. Older statuses are historical.

# Current action: side-network design selected — 2026-10-25

Preserve the frozen full backbone and train a small residual side network from
intermediate features. Artificial CPU/CUDA structural checks passed; cost and
task quality have not been measured. Side/Ladder/LAST are prior art; no novelty
claim. Next: bounded operation-cost feasibility before freezing a real screen.
No new real training command. Stage14C remains a failed practical recipe with
a small directional reset effect. Master v7.19; paper memory v9.
Design: protocol/strategy/SIDE_ADAPTATION_DESIGN_v1.md.
Prior evidence preserved; publication on hold. Older statuses are historical.

# Current decision: Stage14C reviewed — 2026-10-24

Reset improves similarity by 0.06 / 0.12 points; interactions +0.08 / +0.14.
This passes the directional check, but remaining deficits 0.74 / 0.14 points
fail the two-seed 0.30-point quality rule. Adaptation time is 19.99% / 18.84%
lower at those qualities. Both full references qualify; saved logits reproduce.
Retain the small methodological finding; park this practical recipe. No new
run command or confirmatory study prepared. Return to distinct design discussion.
Master v7.18; paper memory v8; review results/stage14C_review_v1/REPORT.md.
All prior evidence preserved; publication on hold. Older statuses are historical.

# Current action: Stage14C optimizer-history diagnostic — 2026-10-24

Run scripts/run_stage14C_optimizer_history.cmd (README_STAGE14C.md).
Twelve trajectories: full/spatial/similarity x retained/fresh AdamW, two source
states, five fixed epochs. Output results/stage14C_optimizer_history_v1.
Primary endpoint: quality within 0.30 points and >=10% complete adaptation-time
saving versus strong ordinary controls in both seeds. Mechanism interaction
evaluated separately; memory secondary. No automatic follow-up or new claim.
Artificial CPU/CUDA validation passed; no real Stage14C execution in preparation.
Master v7.17; paper memory v7. Prior evidence preserved; publication on hold.
Earlier statuses and commands below are historical.

# Current action: candidate and hypothesis selected — 2026-10-24

Checkpoint adaptation is the accepted primary objective; inference secondary.
Next proposed diagnostic: similarity merging with retained versus fresh AdamW
state, with matched full and spatial controls. Test compression-specific
recovery and practical complete adaptation cost separately. No new run command.
This is a proposed mechanism study, not an established cause or novel method.

Current master: adaptive_gradient_refresh_project_master_v7.16.md. Paper memory v7.
Design: protocol/strategy/TOKEN_OPTIMIZER_HYPOTHESIS_v1.md.
No new training/scoring; all previous failed screens preserved. Publication on hold.
Earlier statuses and commands below are historical.

# Current action: objective comparison complete — 2026-10-24

Step 1 compares training from scratch, existing-checkpoint adaptation and
inference. Recommended primary objective: complete marginal adaptation cost
at a predefined quality requirement; inference secondary. Recommendation
awaits the author's selection; no candidate or new experiment is selected.
Retain the observed 1.1% source-inclusive token saving alongside its quality
tradeoff. All prior frozen outcomes remain unchanged; publication on hold.

Current master: adaptive_gradient_refresh_project_master_v7.15.md.
Comparison: results/research_objective_comparison_v1/REPORT.md. Paper memory v6.
Next step: select one candidate/hypothesis under the chosen objective.
No new run command. Earlier statuses and commands below are historical.

# Current decision: Stage14B reviewed; fixed recipe parked — 2026-10-24

The recovery run and independent audit are complete. Full: 84.72% / 83.90%;
similarity: 83.92% / 83.64%. Similarity fails the two-seed 0.30-point margin;
spatial also fails. Both full references qualify. All saved logits reproduce.
Recovery savings of 21–23% become about 1.1% after original training cost;
no matched-quality total training-speedup claim. Do not extend this run.

Current master: adaptive_gradient_refresh_project_master_v7.14.md.
Review: results/stage14B_review_v1/REPORT.md. Paper memory: v5.
Return to distinct mechanism design; no new protocol or run command prepared.
Prior evidence preserved. Publication on hold. Older commands below are historical.

# Current action: Stage14B bounded token recovery — 2026-10-24

Run scripts/run_stage14B_token_recovery.cmd. Six runs, five fixed epochs each,
restored AdamW, unchanged midpoint compression. Output results/stage14B_token_recovery_v1.
See README_STAGE14B.md, protocol/stage14b/TOKEN_RECOVERY_v1.md and master v7.13.
Artificial CPU/CUDA validation passed; no real Stage14B training in preparation.
Original source and recovery costs remain explicit. Both seeds must pass the
frozen final quality/resource screen; no automatic tuning or continuation.
Prior evidence preserved; publication on hold. Older commands below are historical.

# Current research state: both experiments reviewed — 2026-10-24

Stage13C has a qualified full reference but the local recipe fails quality
and resource criteria. Stage14A similarity merging saves 21.52% / 23.52%
forward/backward time, with 0.30 / 0.60-point accuracy losses; its both-seed
quality rule fails. These are operation savings, not a demonstrated training
speedup. Both independent integrity audits pass and all rescored logits match.

Current master: adaptive_gradient_refresh_project_master_v7.12.md.
Review: results/stage13C_stage14A_review_v1/REPORT.md. Paper memory: v4.
Recommended next: a separately specified bounded recovery test for similarity
merging. No new protocol or command yet. Park the tested local-loss recipe.
Publication stays on hold. Earlier statuses and commands below are historical.

# Current action: Stage13C matched local trajectories — 2026-10-23

Run scripts/run_stage13C_local_trajectory.cmd. Compare full, connected auxiliary,
local and suffix training over fixed352-update trajectories, two source states.
Restore classifier/auxiliary optimizer histories; charge auxiliary calibration.
Output results/stage13C_local_trajectory_v1. See README_STAGE13C.md and master v7.11.
Artificial CPU/CUDA validation passed; no real Stage13C training in preparation.
All prior evidence preserved; Stage14 queued; publication on hold.
Earlier statuses and commands are historical.

# Current research state: Stage13B reference qualified — 2026-10-23

Both full continuations reproduced all704 original training losses exactly and
improved after352updates. All four arms initially worsened; one-step positivity
was unsuitable here. Saved source/final logits reproduced exactly. Local-learning
utility remains unresolved; next is a separately specified multi-step comparison.
No new run command yet. Stage14 stays queued; publication on hold.
See master v7.10, results/stage13B_review_v1/REPORT.md and paper memory v3.
Prior files preserved. Earlier statuses and commands are historical.

# Current priority: Stage13B reference repair — 2026-10-23

Run scripts/run_stage13B_reference.cmd first. Compare exact optimizer continuation
against reset state over the original fixed epoch and qualify the reference.
Output results/stage13B_reference_v1; see README_STAGE13B.md and master v7.9.
Local-learning utility remains unresolved; Stage14 is prepared but queued.
Paper discussion memory: paper/RESEARCH_DISCUSSION_MEMORY_v2.md. All original
studies preserved. CPU/CUDA artificial validation passed; no real Stage13B run
executed in preparation. Publication on hold. Earlier statuses are historical.

# Current action: Idea4 / Stage14A prepared — 2026-10-23

Intermediate token compression diagnostic prepared and frozen. Compare full
processing, spatial averaging and similarity merging at existing ViT states.
No optimizer updates. Run scripts/run_stage14A_token_diagnostic.cmd; review
results/stage14A_token_diagnostic_v1. See README_STAGE14.md and master v7.8.
Paper discussion memory: paper/RESEARCH_DISCUSSION_MEMORY_v1.md. Retain local
resource shortfalls and inconclusive learning utility without refuting prior
papers. CPU/CUDA artificial validation passed; no real Stage14 diagnostic run.
Prior science preserved; publication on hold. Earlier statuses are historical.

# Current research state: Stage13A reviewed — 2026-10-23

Local-objective integrity audit passed. Utility is inconclusive: the full
reference and every other measured update worsen score loss. Local resource
savings2.41% /9.36%time and6.38% /6.26%incremental memory miss frozen thresholds.
Saved first-trial losses reproduced. Park this design; no long pilot or retuning.
See adaptive_gradient_refresh_project_master_v7.7.md and
results/stage13A_review_v1/REPORT.md. Next queued idea: feature/token compression,
not yet prepared. No new run command. Prior results retained; publication on hold.
Earlier statuses and run instructions below are historical.

# Current action: Idea3 / Stage13A prepared — 2026-10-22

The author approved local objectives / decomposed credit assignment. Stage13A
is prepared and frozen: three blocks, four controls, two saved source states,
96measured disposable updates plus local-head calibration. Run
scripts/run_stage13A_local_diagnostic.cmd; review results/stage13A_local_diagnostic_v1.
See README_STAGE13.md and adaptive_gradient_refresh_project_master_v7.6.md.
CPU/CUDA tests and artificial workflows passed. No real diagnostic executed
during preparation. Prior results retained; publication remains on hold.
All earlier statuses and run instructions below are historical.

# Current research state: Stage12A reviewed — 2026-10-22

The pretrained periodic-update screen is complete:39–40% faster than full but
0.96–1.52pp lower accuracy, failing the frozen0.30pp rule in both seeds.
Baselines were ready. Useful partial recovery and LP-FT control savings retained.
All42,240updates audited and all ten final checkpoint accuracies reproduced.
See adaptive_gradient_refresh_project_master_v7.5.md and
results/stage12A_review_v1/REPORT.md. No automatic tuning or extra epochs.
Next queued direction: local objectives / decomposed credit assignment, not yet
prepared. No new run command. Publication remains on hold.
Earlier statuses and run instructions below are historical.

# Current action: Idea2 / Stage12A prepared — 2026-10-22

The author approved pretrained adaptation with selective updates. Stage12A is
prepared and frozen: ImageNet ResNet-18, five controls, two seeds,12epochs.
Run scripts/run_stage12A_adaptation.cmd and review results/stage12A_adaptation_v1.
See README_STAGE12.md and adaptive_gradient_refresh_project_master_v7.4.md.
Growth's useful time/accuracy observations remain preserved with its failed
both-seed quality rule. Publication remains on hold. CPU/CUDA tests and artificial
workflows passed; no real transfer training executed during preparation.
All earlier statuses and run instructions below are historical.

# Current research state: Stage11B reviewed — 2026-10-22

Stage11B is complete and fails the frozen both-seed quality rule. Growth saved
22–24% time, but accuracy deficits were1.40 /0.12pp against an allowed0.20pp.
Ten runs /176,000 updates audited; all final checkpoint accuracies reproduced.
See adaptive_gradient_refresh_project_master_v7.3.md and
results/stage11B_review_v1/REPORT.md. Close this fixed midpoint recipe; no automatic
extension or tuning. Next queued idea: pretrained adaptation/selective updates,
not yet prepared. No new run command. Publication remains on hold.
Earlier status and run instructions below are historical.

# Current direction: Stage11B growth pilot — 2026-10-22

Stage11A feasibility passed: no changed predictions in10,000 checks; narrow
steps35–45% cheaper. Efficacy remains untested. Stage11B is prepared and frozen:
two fresh seeds, five arms,100epochs, fixed growth after50 with Adam-reset controls.
See README_STAGE11B.md and adaptive_gradient_refresh_project_master_v7.2.md.
Run scripts/run_stage11B_growth_pilot.cmd; then review
results/stage11B_growth_pilot_v1. CPU/CUDA artificial validation passed.
Publication stays on hold. All earlier statuses/commands below are historical.

# Current direction - 2026-10-22

The author approved exploring small-to-large model growth first. The new short
mechanics/cost diagnostic is ready; see `README_STAGE11.md` and project master
v7.1. Earlier completed Stage9/10 statuses and commands below are historical.

# Current research state - 2026-10-21

## Stage10A completed: coarse/detail design rejected

All ten 100-epoch runs and 176,000 updates passed independent audit. Saved
final checkpoint accuracies reproduced exactly on the original training-only
validation split; no training or test evaluation occurred during review.

Mean final accuracy: full RGB32 82.15%, narrow RGB32 81.43%, average RGB16
74.96%, pixel branches 74.73%, Haar branches 74.84%. Both full baselines
sustained the primary 80% target. None of the low-resolution arms did.
Haar failed practical, detail and structure flags in both seeds.

Narrow RGB32 reached the target 40.10% / 21.62% sooner than full, but endpoint
accuracy losses of 0.12 / 1.32 pp miss the desired two-seed quality tolerance.
It is an ordinary control and a useful design clue, not a novel-method win.
This allocation failed to turn lossless coefficients into useful accuracy;
the cause is unresolved. The result does not reject all frequency methods.

Close the tested design and return to mechanism design. No new training
command is needed now. No automatic transform/architecture/optimizer grid.
Publication stays on hold. Raw results, frozen science, prior masters and
manuscript v7 are preserved; 754 immutable files verified in this review.

Current master: `adaptive_gradient_refresh_project_master_v7.0.md`.
Review: `results/stage10A_review_v1/REPORT.md`, `independent_audit.json` and
checkpoint rescoring tables. Decision:
`protocol/stage10/POST_SCREEN_DECISION_v1.md`.
Raw completed data: `results/stage10A_representation_v1/`.
Review reproduction: project Python + `scripts/review_stage10A_results.py
--rescore-cuda` (validation and audit only; no training).

Earlier status sections and commands below are historical, superseded by
the completed reviews. They are not instructions to relaunch training.

## Historical status: Stage9E failed its practical screen — 2026-10-21

All ten frozen100-epoch development runs completed and176,000 updates passed
review. Final checkpoints were rescored; all ten final accuracies reproduced.
Both adaptive candidates failed speed AND final-accuracy criteria on BOTH seeds.
Full baselines reached the primary sustained80% target, so this was not a
readiness failure. Publication remains on hold.

Mean final accuracy: full81.72%, periodic81.40%, random81.15%, boundary80.19%,
MixUp79.07%. Boundary took about35% longer than the fastest blind control to
target in each seed. Fixed-update savings did not yield useful learning speed.
No joint practical win over full was established across both seeds by any arm.

Read `results/stage9E_review_v1/REPORT.md`, `independent_audit.json` and
`protocol/stage9/POST_PILOT_DECISION_v1.md`. The review also discloses quota-
forced selections and numerical full-warmup drift; this is a development
rejection of the tested recipe, not universal proof about every possible gate.

Current master: `adaptive_gradient_refresh_project_master_v6.8.md`.
Close this controller/cut4/40%-quota branch. No automatic epoch extension,
quota reduction, new cut/architecture/seed or gate normalization. Return to
distinct mechanism design and reproducible comparisons before new GPU work.
There is no new training command to run. Existing results/sources remain intact.

The sections below preserve earlier plans and commands as historical
reproduction instructions. They are superseded by this completed-pilot decision.

## Historical preparation: Stage9E five-arm quality pilot — 2026-10-21

The author approved the training-quality test. Code/protocol are frozen and
CPU/CUDA artificial checks passed. Run in Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage9E_quality_pilot.cmd
```

Stop other GPU jobs first. Two new DEVELOPMENT seeds1301/1302, five arms,
100 epochs each: full AdamW, periodic, random, MixUp-only and boundary-RMS.
One boundary4, identical prefix-freezing semantics and matched omission/example
budgets. All arms use the same data/initialization/random streams per seed.
Five initial full epochs, then40% of regular opportunities omitted subject to
nonadjacency and fixed quotas. Short batches always full. No test dataset.

Primary target:80% validation accuracy sustained for three evaluations. A
promising development result must save >=5% observed time versus full AND the
best reaching predecided control, with final accuracy within0.2pp of each on
BOTH seeds. All controller/calibration costs are counted; no target changes,
automatic extensions or publication claim. See the fixed protocol for details.

Output: `results/stage9E_quality_pilot_v1/`. Send the entire directory when
complete or if an error occurs. Allow roughly0.5--1GB plus incomplete attempts;
1--2 hours is a rough runtime estimate from recent measurements. Rerun the
same command after interruption: complete trajectories verify/skip, incomplete
ones restart in new attempts and old attempts remain. Changed settings/source/
environment block resume explicitly.

Protocol: `protocol/stage9/QUALITY_PILOT_v1.md`; prospective freeze:
`protocol/stage9/FROZEN_QUALITY_PILOT_v1.json`; checks:
`protocol/stage9/VALIDATION_QUALITY_PILOT_v1.md`. Real training was not launched
during preparation. Current authoritative master:
`adaptive_gradient_refresh_project_master_v6.7.md`. Publication stays on hold.

The sections below retain completed diagnostics and their earlier next-step
recommendations as history; the approved frozen Stage9E pilot supersedes them.

## Historical status: Stage9D completed and reviewed

All512 batch256 timing measurements passed independent integrity and summary
checks. Read `results/stage9D_review_v1/REPORT.md` and `audit.json`.
Current authoritative master: `adaptive_gradient_refresh_project_master_v6.7.md`.

At cut4 and hypothetical40% omission, efficient predecided skipping projects
17.34% lower operation cost than full updates; boundary-RMS skipping projects
15.85%. The boundary decision adds1.81% relative cost versus efficient
predecided scheduling. This leaves computational headroom but establishes no
training-quality or time-to-quality advantage. The earlier post hoc cheap
MixUp comparison prevents an incremental-utility claim for the gradient gate.

Do not rerun completed timing studies or start a boundary grid. If continuing,
next is a bounded single-cut4 training-quality pilot with full, periodic,
matched random, MixUp-only and boundary-RMS controls under matched freezing
semantics/resources. Freeze online rules, targets and stopping before results.
Such a pilot is proposed, not yet implemented/frozen/launched. Publication
remains on hold. Current completed-review reproduction (no training):

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -B scripts\review_stage9D.py
```

The launch commands below are retained for historical reproduction/resume.

## Completed and audited: Stage9C

The real run is complete: all6,144 branch measurements and eight saved states
passed independent review. Read `results/stage9C_review_v1/REPORT.md`.
Current authoritative master: `adaptive_gradient_refresh_project_master_v6.7.md`.

Larger prefixes have computational headroom: at hypothetical40% omission,
cut4 projects10.2% (zero) /12.7% (freeze) savings, versus3.4% /4.7% for old
cut2. This is not measured end-to-end training improvement. Boundary RMS
predicts omitted-update error, but an explicitly post hoc MixUp-strength
comparison captures most of that information; additional validation value
is unproven. No replacement controller has been trained.

Do not rerun completed training or start a large sweep. Next is a deployable
batch256 cost check against efficient blind and cheap paths. If justified,
a bounded pilot should fix one cut (development candidate:4), use the same
prefix-freezing semantics, and include full, periodic, matched random,
MixUp-only and boundary-gradient controls. Freeze online rules and quality/
time targets prospectively. Publication stays on hold.

The completed Stage9D reproduction command is a COST-ONLY batch256 comparison at existing
checkpoints15/30, cut4. It compares ordinary full updates, efficient predecided
freezing, and boundary-RMS full/partial paths. No trajectory is retrained:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage9D_deployment_cost.cmd
```

Stop other GPU work first. Send the entire `results/stage9D_deployment_cost_v1`
folder when complete. CPU artificial mechanics checks passed; actual CUDA
paired checks run before measurements. Protocol and validation:
`protocol/stage9/DEPLOYMENT_COST_v1.md` and
`protocol/stage9/VALIDATION_DEPLOYMENT_COST_v1.md`. The author executed Stage9D after preparation; it is now audited. This
benchmark cannot establish accuracy gains.

The original reproduction/resume command is retained below (completed blocks
skip; no new efficacy experiment):

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage9C_boundary_diagnostic.cmd
```

Stop other substantial GPU jobs first. This collects TWO full-AdamW
development trajectories, seeds1201/1202,30 epochs each. It then compares
six cut points at the SAME model/optimizer checkpoints (epochs1,5,15,30):
tokenization only, or tokenization plus the first1,2,3,4,5 ViT blocks.
The old two-block prefix is included. This is the six-block ViT diagnostic;
it does not train six independent boundary-specific models.

At each cut it measures boundary signals, actual one-step update error,
separate-validation loss effects and measured operation costs for explicit
zero prefix gradients versus prefix freezing.64 probes per state separate
calibration from scoring; all cuts share each probe's data and dropout.
The full run produces6144 branch measurements. No test dataset is loaded.

The learning-rate schedule is the FIRST30 epochs of the existing300-epoch
recipe, not a newly compressed30-epoch schedule. Early checkpoint accuracy
is not judged against the prior300-epoch readiness threshold. This is a
mechanism/cost diagnostic, not a new controller efficacy study.

Output: `results/stage9C_boundary_diagnostic_v1/`. Retain the entire folder
and tell the assistant when complete. Review `REPORT.md`, shared checkpoints,
`signal_comparisons.csv`, `projected_costs.csv` and all raw measurements.
Cost projections are not established training speedups; the code does not
automatically pick or promote a winning boundary.

Rerun the same command after interruption. Matching complete training/probe
blocks skip; incomplete blocks restart in new attempts while old attempts
remain. Changed source/settings/environment fail explicitly. Allow roughly
0.5--1GB for checkpoints/logs, more if interrupted attempts accumulate.

Protocol: `protocol/stage9/BOUNDARY_DIAGNOSTIC_v1.md`; validation:
`protocol/stage9/VALIDATION_BOUNDARY_v1.md`. Optional artificial CPU rehearsal:
`scripts/run_stage9_00_boundary_sanity.cmd`. Real scientific collection was launched by the author and is now complete.
The preparation record remains preserved. Current master:v6.7.

## Retained reset rationale and completed preliminary work

Publication preparation is on hold at the author's request. The objective is
a useful contribution beyond periodic/random skipping, not a better-looking
version of the same failure. Prior studies and manuscript v7 remain preserved.

Plan and evidence: `protocol/stage9/RESEARCH_RESET_v1.md`.

Completed without new training:

- Exact target-entropy reconstruction on four existing development trajectories
  (211,200 updates). Raw loss gating strongly follows mixing strength; subtracting
  target entropy only partly reduces this selection. Offline shadow masks do
  not establish a trained policy's accuracy or cost.
- Artificial CPU proof that a native ViT can compute exact suffix gradients,
  inspect its boundary error signal, and then selectively continue prefix
  backward. Ordinary full gradients are reproduced exactly when it continues.

Stage9C above is complete. Do not launch another24-run efficacy study. A new
candidate must earn an advantage over full, periodic and matched-random
baselines; changing the gate formula alone is not established novelty.

Optional reproduction of the completed CPU analyses in Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
"%PYTHON%" -B scripts\stage9_entropy_diagnostic.py
"%PYTHON%" -B scripts\stage9_split_backward_proof.py
```

These are diagnostics/proofs, not new training or performance validation.
Current authoritative master: `adaptive_gradient_refresh_project_master_v6.7.md`.
