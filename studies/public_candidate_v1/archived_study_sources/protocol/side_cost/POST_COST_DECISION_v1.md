# Side-network cost decision — 2026-10-25

The author requested the bounded timing feasibility check. The unchanged
multi-depth side prototype PASSES all four predeclared operation-cost conditions. The gate required median paired update
time <=90% of full in each of two artificial seeds and both batch sizes (256/200).
Eight counterbalanced/interleaved rounds were measured per seed/batch. Full,
head-only and both side variants were rerun in each round; no size/tap search.

| Artificial seed | Batch | Multi-side update ms | Paired saving vs full | Passing rounds / 8 |
|---|---:|---:|---:|---:|
| 99031 | 256 | 19.605 | 33.97% | 8 / 8 |
| 99031 | 200 | 17.143 | 26.01% | 8 / 8 |
| 99032 | 256 | 20.218 | 31.43% | 8 / 8 |
| 99032 | 200 | 16.979 | 26.19% | 8 / 8 |

Prepare and freeze the ten-run real quality screen next. This check does not launch real training.

This is an artificial operation-cost result on the current GPU, not a real-data
adaptation outcome. Every measured update includes native backbone forward,
side/feature-capture overhead where applicable, smoothed loss/scalar readback,
backward and AdamW. Separate tables retain model construction/transfer, discarded
warmup/restoration, fresh first-state allocation and CUDA memory. The operation
clock excludes shared input creation/transfer, augmentation, validation, logging
and checkpoint saving. Setup is fresh-model setup in an initialized process,
not system cold-start latency. No complete adaptation speed claim follows.

All 128 trials, 384 discarded warmup updates and 1,152 retained artificial updates
were verified. Source/optimizer restoration and frozen-backbone checks passed;
all trainable gradients were finite. Both side architectures retain 65 tokens
and have 125,002 trainable parameters. Head/final-side task quality is unknown,
so no comparison with the fastest quality-qualified control is established.
Repeated timing observations are not independent learning/quality replications.

The real task still requires quality within 0.30 points and >=10% lower complete
adaptation cost against strong controls in both source states, if a study is
prepared later. Artificial timing cannot satisfy that screen or establish
novelty. No real checkpoint/data was loaded and no real task scoring or training
occurred. The side-network family is established prior art. Earlier failures
and all prior evidence remain intact; publication remains on hold.

See results/side_cost_v1/REPORT.md, summary.csv and paired.csv, plus
protocol/side_cost/COST_PROTOCOL_v1.md and the frozen cost manifest.
