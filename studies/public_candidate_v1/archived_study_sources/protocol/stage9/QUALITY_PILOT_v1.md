# Stage9E bounded quality pilot — 2026-10-21

Author-approved follow-up to Stage9C/D. Cut4 has cost headroom; boundary RMS's
incremental information over cheap MixUp is unproven. Test learning quality
instead of fitting more diagnostic gates. This is development on the previously
reused45k/5k training-derived CIFAR10 split, not independent validation. No test
dataset construction or reuse of the previous test results for policy tuning.

## Fixed budget and recipe

Two new DEVELOPMENT run seeds1301/1302, five arms each,100 epochs maximum:
full AdamW; efficient periodic; matched random; MixUp-only; boundary-RMS.
Reverse run orders across seeds mitigate order drift; two seeds cannot balance
all five positions or support confirmatory uncertainty estimates. No automatic
additional seeds, cut search,100-to300 extension or recipe repair.

Use the native six-block ViT and Stage8v2 recipe: LR.001, AdamW wd.05,
MixUp alpha.2, label smoothing.1, batch256, five-epoch LR warmup. Train the
FIRST100 epochs of the300-epoch cosine horizon, not a compressed100-epoch
schedule. All arms share initialization, minibatches, augmentation, MixUp and
dropout streams per seed. Fresh suffix gradients on every update.

The first five epochs are full updates. Thereafter freeze tokenization and the
first four blocks on partial updates; suffix retains two blocks, final norm
and head. Prefix grad=None: parameter, moments, step counter and decay all
stay unchanged. Use the SAME semantics in every skipping arm. No zero-gradient
continuation or replay ablation in this pilot.

## Exact resource matching with online decisions

Every epoch has175 full-size256 batches and a final200-example batch. The
short batch is ALWAYS full and does not enter gate calibration. This avoids
matching calls while omitting different numbers of examples. Partition each
epoch's175 regular opportunities into windows20,20,...,20,15; each window
omits round(.4*n), hence8 in a20-window and6 in the15-window.70 omissions
per active epoch,6,650 per partial arm and1,702,400 omitted prefix examples.
All arms still execute17,600 suffix updates; partial-arm prefix counters end
at10,950. No consecutive partial updates, including across windows/epochs.

Periodic uses evenly spaced positions floor((j+.5)*n/k) in each window.
Random samples an exact-count nonadjacent mask online: at each position its
skip probability is the number of feasible skip continuations divided by all
feasible continuations. For n remaining positions, k remaining omissions and
previous-partial p, the continuation count is C(n-k+1-p,k). Independent local
random stream; no minibatch/loss information. Draw once per opportunity,
including forced choices, so the mask can be reconstructed exactly.

MixUp-only prefers high min(lambda,1-lambda), implemented as a low negative-
strength signal. Boundary-RMS prefers low RMS of the fresh bridge gradient
after suffix backward. Each compares its current signal with the40th percentile
of the preceding128 REGULAR-batch signals. Collect calibration signals on the
fifth, full-update epoch; keep a rolling past-only history afterward. Calculate
threshold BEFORE adding the current signal. No score-window sorting, validation
labels, entropy fit or future features enter a decision.

Quota safety is shared: refresh after a partial; stop omitting when quota is
used; force omission only when choosing full would make the remaining quota
infeasible. Otherwise follow the signal, periodic plan or random draw. Record
requests, thresholds, histories, remaining quota and forced-decision reasons.
Small windows constrain adaptation deliberately to match resources exactly;
this is not a claim that the constraint is optimal.

Predecided partial arms run the prefix forward under no_grad. Their full
updates use ordinary native forward/backward. Boundary-RMS builds the prefix
graph, computes suffix backward/RMS, then continues prefix backward only on
refresh. It pays for RMS on all regular active/calibration updates to maintain
history; short batches use ordinary full updates. Calibration and controller
cost belong inside training time.

## Timing, observations and practical screen

Exclude16 disposable hardware warmup operations, exercising all four kernels;
then restore initialization, fresh Adam state and all RNG streams. One epoch
start/end synchronization for active-loop timing, shared loss.item each update;
boundary scalar extraction is additional. No per-component benchmark fences.
Log each epoch's actual active-loop time separately.

Validate hard-label CE/accuracy on the fixed5k split after every epoch.
Primary quality target80%, sustained for THREE consecutive epoch evaluations;
time is when the THIRD qualifying validation completes. Secondary82% and first
crossings are descriptive. No target or readiness threshold changes afterward.
If full baselines do not reach80% in100 epochs, the screen lacks readiness;
keep all results and stop without lowering the target or extending training.

Primary observed time runs from training start through the confirming
validation and includes calibration, decisions, trace writes, prior epoch
logging and scheduled telemetry. Excludes common hardware warmup/model/data
setup and final checkpoint writing. Report active-loop, validation, hardware-
warmup, checkpoint and experiment elapsed clocks separately. Do not treat a
never-reached target as a fast run or silently drop it from summaries.

For each named candidate (MixUp-only and boundary-RMS), a promising development
screen requires BOTH seeds: reach80%; >=5% shorter observed time to target than
full; >=5% shorter than the fastest target-reaching predecided control; and
final accuracy within0.2 percentage points of BOTH full and that control.
MixUp's control pool is periodic/random. Boundary's pool also includes MixUp.
Retain every control's attainment and accuracy; if no control reaches target,
the automatic joint screen is unavailable rather than a vacuous win. Missing
candidate/full targets fail the automatic screen. This is a conservative
development decision, not a two-seed noninferiority or superiority test.

Do not automatically select a seed/epoch/cut or claim novelty. A promising
screen still needs independent confirmation, another workload and close prior-
art comparison. Failure should prompt redesign, not another normalization grid.

## Integrity and interruption

Freeze spec, scientific code, tests, launcher and input-evidence hashes BEFORE
real execution in FROZEN_QUALITY_PILOT_v1.json. Sources/environment/settings
must match exactly. Completed run hashes are verified before skipping. An
interrupted trajectory restarts from initialization in a NEW attempt directory;
completed trajectories skip and incomplete attempts remain. Epoch traces are
flushed to disk. No silent mid-run checkpoint continuation or deletion.

Final checkpoints include model/Adam state, counters, controller/history and
RNG streams. Audit hashes, shared streams, quota feasibility and past-only
thresholds, masks, fixed LR, totals, targets and checkpoint counters before
reporting the pilot. Artificial CPU/CUDA rehearsals use separate outputs and
have no scientific interpretation. Numerical or integrity failures stop the
launcher; no alternative recipe follows automatically.
