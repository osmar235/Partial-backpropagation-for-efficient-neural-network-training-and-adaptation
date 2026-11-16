# Prefix adaptation — reviewed results v1

Both partial-update arms fail the predeclared joint screen. Periodic cut3/r4
saves20.84%/19.69% complete time, but loses0.44/0.36 accuracy points at the second
checkpoint against the allowed0.30. Frozen cut3 loses more quality on both.
Read REPORT.md for all scores, timing, interpretation and preserved failures.

Reproduce grouped statistics using standard Python from this folder:

```cmd
python -B analysis\summarize_results_v1.py review_data.json --output reproduced_groups.json
```

Use a fresh output path; compare with group_summary.json. This command was
executed during packaging and reproduced JSON exactly; no torch/GPU needed.
review_data includes36 raw summaries and24 paired comparisons. Saved score table
contains252 independently recomputed validation-array results. Frozen audit
re-evaluated all36 final checkpoints in both modes, with72 exact rescores.
Historical full baseline weights/moments reproduce exactly in all12 runs.

Full checkpoint/logit files and datasets remain at original project paths;
this addendum contains derived evidence and reproducible aggregation. Frozen
source capsule: ../prefix_adaptation_addendum_v1. Internal review/preservation
helpers stay under protocol/prefix_adaptation. Source execution retains local
absolute provenance and still needs separate portability work before release.

No training during review, no automatic rescue sweep, no next run prepared and
no external publication. Keep full BF16/automatic as the qualified baseline.
