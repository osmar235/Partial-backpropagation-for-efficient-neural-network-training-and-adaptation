# Sampled weight-gradient cost decision — 2026-10-25

The frozen sampled-weight-gradient implementation FAILS the artificial
operation-cost gate in all four seed/batch conditions. It is slower than native
backpropagation by median paired 4.97% / 13.82% / 5.56% / 11.24%, in the order
shown below. None of 24 paired candidate rounds achieves the required 10% saving.

| Artificial seed | Batch | Native update ms | Sampled update ms | Paired slowdown | Passing rounds / 6 |
|---|---:|---:|---:|---:|---:|
| 99051 | 256 | 28.843 | 30.236 | 4.97% | 0 / 6 |
| 99051 | 200 | 22.252 | 25.485 | 13.82% | 0 / 6 |
| 99052 | 256 | 28.820 | 30.349 | 5.56% | 0 / 6 |
| 99052 | 200 | 22.637 | 25.552 | 11.24% | 0 / 6 |

The exact-custom control is approximately 18.75–20.13% slower than native.
Sampling lowers the custom implementation's measured time by approximately
7.81–12.10% (median paired comparisons), but does not overcome its overhead.
This supports an implementation-overhead interpretation, without identifying
which kernels, gathers, RNG operations or dispatch costs dominate. The control
includes all-token index generation/gathering and custom-autograd execution;
it does not isolate Python overhead alone. Fewer arithmetic terms did not
produce faster native-relative updates on this ViT/hardware/configuration.

Incremental steady-update allocated memory is about 15.24–15.52% lower
than native (median paired). Retain that resource tradeoff; memory cannot replace
the predeclared primary time gate. No real quality, complete adaptation, inference
or larger-model result follows. Initial matching, unbiasedness unit checks and
valid gradients establish implementation properties, not task-level usefulness.

Independent review reran the frozen read-only audit and recomputed every paired
ratio and decision from raw trials. Verified 72 trials, 648 retained artificial
updates and 216 discarded warmups; inputs, initial states, first losses and
counterbalanced order agree. Timing repetitions are not independent training
replications. All prior and raw evidence is unchanged. This review performed
no new benchmark, optimization updates, real checkpoint access or task scoring.

Decision: park this fixed quarter-patch implementation for the time-efficiency
objective. Do not launch the proposed real-data accuracy study, change the
sampling fraction from these timings, weaken the 10% criterion or compare only
against the slower exact-custom control. The result does not refute sampled
backpropagation generally, its prior literature, or a future independently
justified fused implementation. Any such redesign requires a separate explicit
design and new prospective cost check. No replacement experiment is prepared or
launched. Preserve this negative systems result in the paper memory; publication
remains on hold.
