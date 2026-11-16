# Attention/copy profile — reviewed evidence v1

Audits pass; no reliable equivalent-method speedup is established. Explicit forward
copy is about 0.5% of linked profiled GPU time. Ordinary timings have large reversals;
normal-dropout learning differs across layouts. Read REPORT.md for full scope.

Reproduce descriptive grouped statistics with standard Python from this folder:

```cmd
python -B analysis\summarize_results_v1.py review_data.json --output reproduced_groups.json
```

Use a new output path and compare with group_summary.json. Packaging ran this
command and verified exact JSON reproduction. No torch, GPU or training is needed.
This reproduces aggregation of supplied evidence, not independent raw-trace parsing
or the frozen experiment audit. The separate trace reconstruction is explicitly
post hoc; do not mistake frozen `outside` for non-backward overhead.

Raw trace/measurement files remain at original project paths identified in
RAW_EVIDENCE_SHA256.json. Review data includes all 32 pairs, 16 profile summaries
and 16 corrected trace reconstructions; raw_measurements contains 64 arm records.
The first draft reconstruction is internal history and is excluded here.

Frozen experiment code/protocol: ../attention_copy_profile_addendum_v1.
Internal audit and preservation scripts: protocol/attention_copy_profile in the
project. Full source execution still needs path portability work before release.
No external publication. No new scientific run or next run command.
