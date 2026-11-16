# Optimized prefix/suffix cost — reviewed results v1

Read REPORT.md for the complete results, comparator sensitivity and limitations.
All cuts 3/5/6 retain descriptive cost headroom against both full controls;
timing is noisy and no accuracy or complete adaptation benefit is established.
The earlier Stage9E quality failure remains closed and part of the interpretation.

From this folder, reproduce the grouped statistics with standard Python:

```cmd
python -B analysis\summarize_results_v1.py review_data.json --output reproduced_groups.json
```

Use a fresh output path; compare with group_summary.json. No torch or GPU needed.
The portable analysis was executed during packaging and reproduced JSON exactly.
review_data.json contains all 112 raw summaries and component records; provenance
and original raw traces/checks remain in the project results. Raw checkpoints and
data are not bundled. Runtime state checks are not independent checkpoint replay.
telemetry.json contains session-boundary samples, not continuous measurements.

Frozen runner/source capsule: ../optimized_prefix_cost_addendum_v1. Internal
review/preservation scripts remain under protocol/optimized_prefix_cost. This
saved-data analysis is portable; the experiment source retains absolute local
provenance and needs separate portability work before external release.
No external publishing and no next experiment launched.
