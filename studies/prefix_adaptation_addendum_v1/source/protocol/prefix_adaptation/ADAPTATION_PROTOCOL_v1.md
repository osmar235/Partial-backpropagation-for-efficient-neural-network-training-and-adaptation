# Prefix adaptation quality and complete cost — 2026-10-28

Prospective follow-up approved after the optimized cost screen. The single
periodic candidate (cut3/r4) was selected after reviewing cost results as a
conservative learning hypothesis, not as a preregistered or optimal winner.
Earlier Stage9E failures and all previous gates remain unchanged. This study
tests restricted adaptation from trained checkpoints, not a rescue of the old
from-scratch RMS/MixUp controller or novelty for conventional scheduling.

## Fixed arms and recipe

Two original archived full ViT source checkpoints, seeds 1501/1502. Six encoder
blocks, original train-mode dropout, pinned reference attention, BF16 forward
under automatic SDPA, explicit FP32 cross-entropy outside autocast. FP32 weights
and moments; no compiler, GradScaler, clipping, coordinate exemption or caching.
All inputs receive a fresh full forward. Same original 45000/5000 training-derived
train/validation split; no test data. These reused sources/validation data make
this an exploratory fixed-checkpoint screen, not independent confirmation.

Arms: full updates; frozen_cut3; periodic_cut3_r4. The prefix is patch projection,
class/position embeddings and first three encoder blocks (1356672 coordinates).
The suffix is the last three blocks, final encoder norm and classifier (1336906
coordinates). Frozen arm never updates prefix. Periodic arm updates prefix at
zero-based global training indices 0,4,8,...,876; no epoch-counter reset. All
880 updates train the suffix, 220 train periodic prefix, zero train frozen prefix.
Inactive prefix gradients are None: no weight decay, moments or counter advance.
No compensation/rescaling for skipped learning rates or AdamW bias correction.
The common learning-rate schedule advances with global updates, not prefix age.

Five epochs, batch 256 (176 batches/epoch, last batch 200), LR 1e-4, one-epoch
warmup then cosine floor ratio .1, AdamW wd .05, MixUp .2, label smoothing .1.
Use the established BF16 adaptation data, augmentation, MixUp, permutation and
dropout seed streams. Optimizer options match source but moments/counters start
empty for all arms. Original optimizer age17600 is not resumed. No cost-screen
calibration anchor or endpoint is used. Three hardware-warmup updates per arm
are discarded, with exact weights/empty optimizer restoration before learning.
Reset parameter flags and routing state as well as tensors; inactive parameters
may not retain warmup gradients or moments. Profile first warmup dispatch only.

Six fresh-process sessions cover all six arm permutations per source; source
order alternates and the second source offsets its permutation by three. Each
arm appears twice at each ordinal position. Fixed streams across sessions are
timing/numerical repeats, not six extra quality seeds. Total36 trajectories,
31680 training updates and108 discarded warmups. No automatic extension or tuning.

## Quality, cost and refresh-value decisions

Common FP32/math validation before training and after every epoch, plus final
own-policy BF16/automatic validation for every arm. Save all logits, curves,
checkpoints, exact streams and per-step schedule/loss records. No best-epoch
selection. Source common-FP32 predictions must reproduce the archived source
within1e-5 with identical predicted classes. The full baseline must finish at
least80% in both scoring modes, be within0.30 points of its source in common
FP32, and within0.30 points of the previously qualified BF16/automatic full
endpoint in each matching mode, in every source/session. Historical full common
accuracies84.58%/83.94%, own-policy84.68%/83.88%; provenance binds raw records.
Report reproduction differences; exact trajectory identity with history is a
diagnostic, not an undisclosed new hard gate.

For EACH partial arm versus paired full: final common-FP32 accuracy and own-policy
accuracy must each be within0.30 percentage points of the full score in that
SAME evaluation mode, in every source/session. Report actual loss and accuracy
deltas, including gains/deficits. This fixed-checkpoint screen is not a population
noninferiority test; repeated identical trajectories do not increase quality n.

Primary clock is COMPLETE adaptation wall time: per-arm data/model/checkpoint
setup, source validation, profiled warmup and restoration, all training including
augmentation/routing/RNG, epoch validation, final own-policy evaluation, final
checkpoint/logits/streams/CSV writing. Imports, idle preflight, frozen-provenance
verification, telemetry after trajectory, post-clock state checks and post-run
checkpoint inference audit are excluded. Summarize components and training-only
times separately. Artifact size can differ across arms because frozen prefix
has no optimizer moments; this is included and disclosed. Source pretraining
cost remains separate; do not claim faster pretraining or add earlier savings.

Require positive paired complete-time saving in all six sessions in BOTH sources,
baseline readiness, both quality views and no session fragmentation. As in the
qualified BF16 study, report exact one-sided sign p per source, maximum p for the
both-source claim, then Bonferroni over the two partial-arm comparisons. Adjusted
p must be<=.05; six/six positive in each source gives .03125. This conditional
direction inference assumes independent session effects; shared-machine drift
limits it. No new minimum5%/10% effect-size floor. Report medians/ranges and every
paired result. Integrity PASS alone does not mean quality/cost qualification.

Also report periodic-minus-frozen accuracy differences in both scoring modes and
periodic/frozen complete-time ratio for every pair. These are descriptive.
Predeclare interpretation:
- If frozen qualifies, periodic refresh is not required to meet this fixed screen;
  report any quality tradeoff, but do not infer a need for an adaptive controller.
- If periodic qualifies and frozen fails the QUALITY margin in every paired
  source/session (at least one evaluation mode in each pair), record that refresh
  was needed for this margin on these checkpoints. This is not population
  superiority, novelty or proof that adaptive scheduling beats a fixed schedule.
- If periodic qualifies but the frozen quality failure is inconsistent, refresh
  necessity is unestablished; preserve all mixed outcomes.
- If periodic fails, close this tested recipe without an automatic cut/R rescue.
  A frozen-prefix success is reported separately as a conventional baseline.

## Integrity, restart and preparation

Audit every per-parameter final AdamW age: full880; frozen suffix880 and prefix
absent; periodic suffix880/prefix220. Verify finite model/moment states, missing
inactive gradients at runtime, and bitwise unchanged frozen prefix weights.
Reconstruct epoch permutations, augmentation RNG usage, MixUp, LR, dropout and
global refresh cadence; compare starting states and streams across arms. Reload
each final checkpoint; rescore common FP32 and own policy with max logit error
<=1e-5 and identical predictions. Record exactness separately; repeated audits
may have different valid rescore errors without changing stored science.

Unexpected/nonfinite errors abort and preserve attempts. Restart skips verified
completed trajectories. Multiple launches or attempts mark a fragmented session
and block practical timing qualification. Completed reruns only audit, including
inference. No silent deletion, tolerance relaxation or favorable-subset analysis.
CPU tiny smoke uses FP32/math but all three schedules; CUDA tiny smoke uses BF16/
automatic. Tiny models cut after block1; those tests cannot establish real quality
or speed. Preparation may inspect/re-evaluate source checkpoints read-only, with
no real training/backward/update. Full experiment remains author-run.
