# Stage11A completed review - 2026-10-22

The growth implementation passes its frozen mechanics and operation-cost screen
in both already-seen source states. Proceed to one bounded efficacy pilot. This
does not yet show faster training to comparable quality or a new method.

| Measurement | Seed1401 | Seed1402 |
|---|---:|---:|
| Narrow / expanded accuracy | 81.92% /81.92% | 80.94% /80.94% |
| Changed predictions out of5,000 | 0 | 0 |
| Maximum absolute logit difference | 4.53e-6 | 4.77e-6 |
| Narrow full-update operation | 19.28ms | 16.04ms |
| Expanded full-update operation | 29.74ms | 28.93ms |
| Ordinary large full-update operation | 29.76ms | 29.05ms |
| Narrow operation-time saving | 35.22% | 44.78% |
| Expansion construction/mapping | 48.41ms | 32.03ms |

The expanded model costs essentially the same per operation as the ordinary
large model. Expansion preserves the small model's existing predictions; it
does not immediately confer the large model's learned accuracy. All48 disposable
trainability updates had finite states and the expected counters, and duplicated
embedding units diverged. This supports trainability, not eventual convergence.

Re-ran the original frozen read-only audit: all26 scientific source files,
four input states, six disposable checkpoints,10,000 validation rows and216
operation measurements passed. Independent arithmetic reconstruction matches
every reported ratio and accuracy.809 immutable inputs verified; all previous
results, sources, checkpoints and masters retained. Current project pointers
will be intentionally updated separately.

The cost experiment resets weights and Adam moments outside operation timers;
it is not a training trajectory or a time-to-quality measurement. Two trained
endpoint states do not establish costs throughout learning. Timing varied across
seeds, so retain the per-seed figures. Strict FP32/math attention also limits
deployment conclusions until an optimized-baseline comparison is performed.

Next: two fresh development seeds, five arms,100epochs, one fixed expansion
after epoch50. Compare uninterrupted small/large models, small/large models with
only an Adam reset at50, and small-to-large with Adam reset at50. The primary
pilot question is complete-run time at comparable final accuracy. Sustained80%
remains a baseline-readiness check and descriptive time-to-target outcome.
This prospective endpoint choice directly measures whether post-growth recovery
consumes the earlier saving; it is set before any pilot result. No old outcome
or threshold is rewritten. Freeze the pilot before execution.
