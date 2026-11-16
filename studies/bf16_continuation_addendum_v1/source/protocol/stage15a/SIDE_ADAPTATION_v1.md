# Stage15A: fixed side-network adaptation screen — 2026-10-25

The author approved continuation after the artificial operation-cost gate.
Freeze this complete recipe before real execution. It is an exploratory screen
on two reused development states, not confirmation or downstream transfer.
No Stage15A real updates or real scoring are permitted during preparation.

## Sources and endpoint

Start from original Stage11B full checkpoints, seeds 1501/1502, age 17,600.
Use the original CIFAR-10 development split: 45,000 training / 5,000 validation;
never access the held-out test set. Five epochs, 176 updates per epoch, 880 per
trajectory; ten trajectories and 8,800 retained updates. Use epoch 5 only.
No best-epoch selection, new seeds, epoch extension or parameter sweep.

## Five conditions

| Mode | Trainable parameters / schedule |
|---|---|
| full | Full original ViT: 2,693,578 parameters |
| head | Classifier only: 1,930 parameters |
| lp_full | Classifier in epoch 1; entire model in epochs 2–5 |
| final_side | Frozen backbone; 125,002 trainable head/side parameters; all ladders read final features |
| multi_side | Candidate; same 125,002 parameters; ladders read blocks 2/4/6 |

Use the unchanged sealed ResidualSideNet prototype: 65 tokens, frozen width-192
six-block ViT, three width-48 side blocks, three attention heads, MLP width 192,
side dropout zero, norm-first encoder layers (default ReLU), LN/projection at
each ladder, CLS normalization and zero output projection 48→192. Shared head
adds the correction with its bias only once. Both side conditions have identical
initial state, using source seed + 280001. The backbone has 2,691,648 parameters.
No feature cache, additional labels, teachers or backbone gradient paths.
The final-feature condition still captures the three taps, then uses the final
map at every ladder: this keeps hook overhead and capacity matched.

Backbone dropout follows train/eval mode in every arm. The native full forward
finishes before side computation. The side output is initially zero; upstream
side gradients must be zero on step 1 and positive from step 2. Initial frozen
logits must match head-only exactly. Against original unfrozen full predictions,
allow maximum absolute logit/loss discrepancy 1e-5, require identical classes
and accuracy. Any failure blocks completion and needs review.

## Optimizer and streams

Fresh AdamW in every arm, preserving the original source group hyperparameters
(betas .9/.999, epsilon 1e-8, weight decay .05); all moments and counters empty.
One canonical parameter group includes inactive parameters. AdamW skips those
with grad=None: no inactive weight decay, state, or step count. This group choice
is fixed here; actual complete timings include its traversal overhead.
In lp_full, enable backbone gradients at epoch 2 without replacing the optimizer:
head moments/counter 176 survive; backbone states start on its first full update.
At the endpoint head age is 880 and backbone age 704. Other active states age 880.
Frozen bodies must remain bitwise unchanged; full/lp_full bodies must change.

Same Stage14C streams: batch +241001, augmentation +242001, MixUp +243001,
permutation +244001, dropout +245001+zero-based-update; crop/flip, MixUp alpha .2,
smoothing .1, batch 256 (last 200). LR 1e-4, one-epoch linear warmup, cosine to
1e-5 over five epochs, no restart when unfreezing. FP32, deterministic settings,
math attention only. Three discarded warmup updates in each assigned initial
condition followed by exact model/fresh-optimizer restoration. Phase 2's first
state allocation and operation are retained and charged, with no extra warmup.
Orders are reversed across seeds as fixed in the JSON specification.

The full trajectory must reproduce Stage14C full_fresh: every saved epoch's
logits within 1e-5 and every training loss within 1e-6. Record all initial/final
states, raw losses, stream hashes, gradients at steps 1/2/176/177/880, optimizer
ages, transition identity and memory. Saved logits are independently rescored.

## Cost accounting

Measure complete instrumented marginal adaptation: source/model setup through
training artifact hashing, including warmup/restoration, transforms, updates,
all six validation passes, phase/gradient/state checks, logging, predictions,
checkpoint save, and an equal one-tenth allocation of shared input preparation.
Source training/run costs stay separately visible. No from-scratch speedup claim.
Memory records distinguish persistent, peak and incremental allocation.
Instrumentation is conservatively charged and may affect this implementation's
cost comparison; report operation and full costs separately.

After this endpoint, separately measure twelve inference batches of 256, using
seed +246001 and three warmups. Post-endpoint inference, offline audit, startup
integrity checks, final summary writing and completion markers are outside the
adaptation clock. Do not describe it as total process latency.

Require one launch for an automatic practical timing decision. Complete arms
are verified and skipped; interrupted arms restart in new attempts, preserving
all old files. Across-launch mixtures retain quality data but withhold the time
decision. Failed/discarded attempts are retained and are not silently claimed as
free realized compute; any campaign-cost interpretation must include them.

## Decision rule, fixed before real results

For each seed, anchor quality to the highest of original full-source accuracy
and the four final control accuracies (full, head, lp_full, final_side). Candidate
must be within 0.30 percentage points. Full must reach at least 80% and remain
within 0.30 points of its original source; at least one control endpoint must
qualify against the anchor. A failed reference calls for review, not a win.

Candidate complete time must be <=0.90 times BOTH full time and the fastest
quality-qualified control time. Final-side belongs to that control set. Require
all reference, quality, time and launch conditions in BOTH seeds. Memory or
inference cannot substitute for the primary time gate. Report the candidate's
quality difference from final-side and whether that control qualifies; do not
attribute improvement to intermediate access when final-side is equally useful.

Artificial outputs always mean pipeline validation, never practical success.
A real pass permits discussing independent confirmation, external-checkpoint
transfer and published baselines; it does not authorize another run. A failure
closes this fixed recipe, not the entire side-tuning literature. Side-Tuning,
Ladder Side-Tuning and LAST remain prior art; this is no exact reproduction and
no new-algorithm claim. All previous failed screens stay intact. Publication hold.

## Preparation validation

Artificial CPU and CUDA checks cover source preservation, native-full matching,
side gradient activation, frozen states, optimizer phase transition, counts,
hook cleanup, complete pipelines/audits, restart and corruption rejection.
Real source mapping is read-only, without data scoring or parameter updates.
Freeze script, protocol/spec, dependency and evidence hashes plus environment.
Preparation records and release pointers are sealed separately after validation.
