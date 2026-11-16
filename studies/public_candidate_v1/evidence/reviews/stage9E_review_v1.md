# Stage9E independent outcome review — 2026-10-21

Both adaptive candidates failed the prospectively frozen practical screen on
BOTH development seeds. The boundary-RMS controller was about35% slower than
the fastest blind control to sustained80% validation accuracy in each seed
and finished with lower accuracy. MixUp-only was worse still. Stop this tested
controller/budget recipe; do not launch a quota, cut or normalization rescue
sweep. Publication remains on hold.

This is a negative development decision, not proof that all selective-backward
methods or boundaries are impossible. No new training was performed in review.

## Verification and preservation

The completed frozen audit was rerun READ ONLY: all10 trajectories and176,000
updates passed. Sources, prospective freeze, configuration, completion hashes,
initialization, final data/MixUp/augmentation/dropout RNG states, past-only
thresholds, feasible masks, quota totals and finite final Adam states/counters
were verified. Every partial arm has6,650 prefix omissions and1,702,400 omitted
prefix examples; suffix17,600 updates, prefix10,950. Full arms have17,600 prefix
updates. Thus different omission counts cannot explain the adaptive/control gap.

The post-run review independently reconstructed sustained targets and candidate
screen calculations from epoch traces. All ten saved checkpoints were reloaded
on CUDA and evaluated again on the unchanged5k training-derived validation
split. Every final accuracy matched exactly and every validation CE matched
within1e-6. No test dataset or training was used for rescore. See
`checkpoint_validation_rescore.csv`, `rescore_metadata.json` and
`independent_audit.json`.

648 immutable files were hash verified before project-pointer updates, including
the completed raw pilot, frozen sources, earlier evidence and preserved masters.
The old scientific runner, audit, protocol, spec and freeze were not edited.
The first optional post-run check had assumed that full-warmup losses would
match numerically; it stopped on observed drift. That extra assumption was
removed from the NEW review script and replaced by an explicit limitation;
the executed study and original criteria remained unchanged.

## Outcomes fixed before these results

Primary target80%, confirmed at the THIRD consecutive qualifying epoch
evaluation.100 epochs maximum, first100 of the300-epoch LR schedule. A named
candidate needed, in BOTH seeds, >=5% shorter observed time than full AND the
fastest reaching predecided control, with final accuracy within0.2pp of each.
Observed time includes calibration, decisions, validation, trace logging and
scheduled telemetry. Missing targets remain censored; no thresholds were moved.

| Arm | Final accuracy seed1301 | Final accuracy seed1302 | Mean final accuracy | Observed seconds to sustained80%, seed1301 / seed1302 |
|---|---:|---:|---:|---|
| Full AdamW | 81.52% | 81.92% | 81.72% | 288.47 /324.35 |
| Periodic | 81.48% | 81.32% | 81.40% | 320.18 /305.42 |
| Random | 81.72% | 80.58% | 81.15% | 302.02 /339.01 |
| MixUp-only | 79.44% | 78.70% | 79.07% | Not reached /415.07 |
| Boundary RMS | 79.76% | 80.62% | 80.19% | 408.77 /413.76 |

The best reaching predecided comparator was random for seed1301 and periodic
for seed1302. For boundary-RMS, including MixUp in the comparator pool did not
change that selection.

| Candidate | Seed | Final delta vs full | Final delta vs fastest control | Time / full | Time / fastest control |
|---|---:|---:|---:|---:|---:|
| Boundary RMS | 1301 | -1.76pp | -1.96pp | 1.4170 | 1.3535 |
| Boundary RMS | 1302 | -1.30pp | -0.70pp | 1.2756 | 1.3547 |
| MixUp-only | 1301 | -2.08pp | -2.28pp | Not reached | Not reached |
| MixUp-only | 1302 | -3.22pp | -2.62pp | 1.2797 | 1.3590 |

Both candidates failed speed AND final-quality conditions in each seed. The
full baselines both reached the primary target, so this is not a readiness
failure blocking interpretation. No arm sustained the secondary82% target
for three evaluations. Secondary failure does not replace the primary test.

Target attainment is not permanent: boundary1301 confirmed80% at epoch96 but
finished below80%; MixUp1302 confirmed at99 and finished78.70%. The final
accuracy guard catches those cases. There is no best-epoch substitution.

## Actual savings at fixed updates did not yield useful acceleration

At100 epochs, mean paired active-training wall ratios to full were:
periodic0.8272, random0.8256, MixUp0.8297 and boundary0.8431. Thus boundary
used about15.7% less training time for the same update count, while needing
substantially more epochs to the quality target. Savings per epoch are not
time-to-quality improvement. The cost headroom from Stage9C/D was real enough
to measure, but did not validate the policy's learning value.

The blind schedules also do not satisfy the desired joint practical result
across both seeds. Periodic1302 reached the target5.84% faster than full but
lost0.60pp at the final epoch. Periodic1301 kept final accuracy close but took
10.99% longer to target. Random took about4.5--4.7% longer than full to target.
There is no useful new-method result to promote from this pilot.

## Allocation and numerical limits — explicitly post hoc observations

Quota matching is an informative but restrictive design choice. Deadline-forced
omissions account for60.40% of boundary skips and65.32% of MixUp skips, versus
19.02% for random and0% for periodic. Under the budget/nonadjacency constraint,
an adaptive signal often could not freely choose whether to omit. This limits
inference about the signal in other policies; it does not change the failure
of the controller that was frozen and actually tested.

Both adaptive policies still selected mixed batches differently from blind
controls. Across the two seeds, omission rates for weak mixing<=.05 versus
strong mixing>=.20 were31.33% /56.22% for boundary and26.97% /60.30% for
MixUp, versus about40% /40% for periodic/random. Associations with mixing
strength do not establish which mechanism caused the accuracy loss. Do not
fit another gate to the same outcome labels and call it independent evidence.

The seeded FULL-update warmup was not numerically identical across arms.
Initial states and logged/final RNG streams matched, and epoch1--3 validation
accuracies matched within each seed. Tiny early loss differences accumulated:
by warmup epoch5 the maximum loss difference from full was0.07305 (seed1301)
and0.03486 (seed1302); the across-arm validation-accuracy ranges were4.42pp
and0.88pp respectively. This occurred even in arms using the ordinary native
full-update path, before any omissions, so it is not uniquely a split-backward
phenomenon. See `initial_full_warmup_differences.csv`.

Strict deterministic algorithms were not required or enabled by the frozen
runner. PyTorch distinguishes seeding/benchmark settings from deterministic
operation requirements; disabling cuDNN benchmarking alone does not guarantee
deterministic computation. [PyTorch2.11 reproducibility documentation](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
Numerical nondeterminism/amplification is a plausible explanation for the
observed drift, not an isolated causal diagnosis. These runs are matched-stream
development comparisons, not bitwise-identical counterfactual trajectories.
Two seeds do not supply an uncertainty estimate establishing the size or
universality of the policy effect. The repeated practical failures justify
not investing in this version; they do not prove a universal impossibility.

Measurement telemetry spanned61--73C and2752--2805MHz. Reversed orders help
with gross drift, but sparse samples and two seeds cannot remove all timing
variation. Endpoint rescoring and trace/resource checks nevertheless support
the recorded outcomes. No claim of a definitive causal mechanism is made.

## Research decision

Close the tested Stage9E branch as failed. Retain the efficient predecided
implementation as a baseline tool, not a demonstrated publishable method.
Do not automatically extend to300 epochs, lower the skip fraction, change cut,
switch architecture, add seeds or normalize the same gate to rescue this result.
Those would be new hypotheses requiring their own reason and prospective plan.

The next phase is drawing-board work: define a learning-value or approximation-
error mechanism with a credible advantage over cheap scheduling, and first
establish a reproducible comparison before allocating another training budget.
Loss or boundary-gradient magnitude alone has not earned that role here.
No replacement method is claimed or commissioned by this review. Preserve
all negative evidence and the frozen history. Publication preparation remains
on hold; current project master:v6.8. No new GPU training command is recommended.

## Reproduction

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\projects\.venv\Scripts\python.exe" -B scripts\review_stage9E_results.py --rescore-cuda
```

This rechecks the completed study and rescores saved endpoints into this
separate review directory. It does not modify the frozen raw study or train.
