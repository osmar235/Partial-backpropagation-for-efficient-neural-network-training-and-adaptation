# Stage9D independent review — 2026-10-20

The cost-only experiment completed correctly. At cut4 and batch256, an
efficient predecided schedule has more projected operation-time savings than
the boundary-gradient path. The gradient path's extra cost is modest, around
1.8% at40% omission, so timing alone does not rule it out. Its additional
learning value remains unestablished; neither path has demonstrated an
end-to-end training or time-to-quality advantage.

## Integrity and interpretation

Audit PASS: all512 expected measurements,128 matched probe batches, four saved
states from two model seeds, all15 executed source hashes, four checkpoint
hashes, configuration fingerprint and all six completion-bound output files.
The checkpoints match the completed Stage9C training markers and development
split. Input-index uniqueness/bounds, matched augmentation/dropout metadata,
balanced path order, paired losses/RMS, positive finite times and timing sums
were verified. All eight original mixture summaries were recomputed from raw
rows without importing the executed analysis functions.

The executed, hash-verified code enforced paired full/partial gradient and
updated-parameter agreement and freeze/counter checks before measurements at
each state. Completion was reached. The independent review did not rerun
those CUDA assertions or reconstruct unsaved gradient tensors.

Verified522 immutable entries from the Stage9C preservation manifest, the
immutable Stage9D preparation files and previous Stage9C review artifacts.
The two mutable project pointers are explicitly tracked separately. Prior
scientific records and manuscripts were not changed. Reproduction:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -B scripts\review_stage9D.py
```

The audit needs only Python's standard library and does not start training.

## Primary measured cost projections

Arithmetic means of four state-level ratios, with augmentation included.
Omission fractions are hypothetical mixtures of forced full/partial operation
measurements, not realized adaptive-controller skip rates.

| Assumed omission fraction | Predecided / full cost | Boundary / full cost | Boundary / predecided cost |
|---:|---:|---:|---:|
| 20% | 0.9133 | 0.9236 | 1.0113 |
| 40% | 0.8266 | 0.8415 | 1.0181 |

At40%, predecided scheduling projects17.34% lower operation cost than full;
the boundary path projects15.85% lower cost. The gap is1.50 percentage points
of the full-update budget. Relative to the predecided mixture, boundary
decisions cost1.81% extra on average (state range1.18--2.43%). At20%, the
respective savings are8.67% and7.64%, with1.13% relative overhead for boundary.

| Seed | Checkpoint | Predecided / full at40% | Boundary / full at40% | Boundary overhead versus predecided |
|---:|---:|---:|---:|---:|
| 1201 | 15 | 0.8379 | 0.8477 | 1.18% |
| 1201 | 30 | 0.8217 | 0.8417 | 2.43% |
| 1202 | 15 | 0.8221 | 0.8352 | 1.59% |
| 1202 | 30 | 0.8246 | 0.8415 | 2.05% |

A predecided schedule can avoid constructing the prefix graph on omitted
updates. The boundary signal needs that graph, suffix backward, a norm reduction
and scalar transfer before deciding. This comparison therefore uses an
efficient blind/cheap kernel rather than forcing those controls through the
adaptive pipeline. It measures their available kernel cost, not the complete
calibration/scheduling/logging overhead of an implemented online controller.

An illustrative fixed-quality calculation: if per-update mixture costs
remained representative, the boundary controller would need approximately
1.78% fewer updates to reach the same quality than predecided scheduling just
to offset its measured40% cost premium (mean of `1 - C_pre/C_boundary`).
This is a break-even calculation, not an observed sample-efficiency gain.
Parameter-update-error reductions cannot be converted into this quantity.

## Timing sensitivity and limits

One first-probe augmentation took83.06ms, compared with sub-millisecond typical
augmentation costs. Augmentation was prepared before operation warmup. Keep
all original samples as primary; do not silently discard that measurement.
Two separately labeled POST HOC checks retain the same interpretation:

| At40% omission | Predecided / full | Boundary / full | Boundary / predecided |
|---|---:|---:|---:|
| All samples, primary | 0.8266 | 0.8415 | 1.0181 |
| Exclude first probe at every state | 0.8231 | 0.8377 | 1.0177 |
| Operation time only, no augmentation | 0.8196 | 0.8351 | 1.0189 |

GPU idle preflight passed. Measurement telemetry samples spanned42--53C and
2812--2835MHz; those sparse samples cannot prove that timing was free of all
hardware effects. Ratios use four states from TWO seeds, not512 independent
model runs. No population confidence interval is inferred from repeated probes.

State restores, model creation, data loading, diagnostic cloning and integrity
checks are outside these operation timers. Inputs and states are repeatedly
reused, and scalar synchronization is included as specified. Cache conditions
and a real training loop can differ. Stage9C and Stage9D differ in batch size,
feature bundle, timing boundaries and forward implementation; their numerical
differences are not an isolated optimization ablation.

## What this resolves, and what it does not

Moving the cut deeper has computational potential. Cut4 is a reasonable fixed
development candidate, not a proven optimal boundary. Efficient blind/cheap
scheduling is a strong baseline. The boundary decision is feasible at modest
additional cost, but has not earned a method-superiority claim.

Stage9C found that much of boundary RMS's selection information was captured by
cheap MixUp strength: about88.8% equal-count subset overlap at cut4; less than1%
incremental update-error reduction; no mean incremental one-step validation-CE
benefit in that post hoc comparison. Those findings neither prove equivalence
nor establish the cheap policy as a useful trained method. In particular, a
1.8% cost premium cannot be justified by relabeling a0.6% parameter-error proxy
gain as learning acceleration.

No adaptive trajectory, training-quality comparison, accuracy improvement,
time-to-quality gain, novelty result or publication readiness was demonstrated
by Stage9D. Publication remains on hold. No new efficacy training was launched
during this review.

## Next research decision

Do not run another timing/boundary grid or fit more gates to these same outcome
labels. The unresolved question is learning quality. If continuing this
direction, the next informative experiment is one bounded single-cut4 quality
pilot with full AdamW, efficient periodic, matched random, cheap MixUp-only and
boundary-RMS scheduling. Use the same freezing semantics for every partial arm;
compare realized resources and include all calibration/gate costs. This is a
proposed development comparison, not a frozen or implemented efficacy study.

Freeze online past-only thresholds, omission budgets, forced-refresh rules,
quality targets, duration and stopping criteria before running. Test a target
that the full baseline can reach within the bounded duration. Retain every
failure and do not lower targets retrospectively. The next stage must test
actual time to comparable quality against the best cheap/blind control, not
just speed against full or a fixed-step proxy. A negative practical screen
should trigger a redesign rather than another normalization sweep.

The earlier proposed >=5% time-to-quality advantage and <=0.2pp final accuracy
loss remain planning values, not a completed or registered hypothesis test.
Only after a practical pilot win would fresh-run confirmation, another workload
and a full novelty review support a publication investment. Current master:v6.6.
