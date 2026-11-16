# Reviewed key-bias functional results v1

Completed 2026-10-27. Read REPORT.md for findings and limitations. Sources and
prospective design remain in ../key_bias_function_addendum_v1 unchanged. This
package adds reviewed JSON evidence and portable descriptive analysis, not a
training implementation, new speed result, quality proof or historical gate waiver.

Reproduce summaries with standard Python (no torch or GPU required), from this folder:

```cmd
python -B analysis\summarize_results_v1.py comparison_rows.json --output reproduced_groups.json
python -B analysis\extract_findings_v1.py comparison_rows.json --output reproduced_findings.json
```

Compare those files with group_summary.json and findings.json. Fresh output paths
are required. During packaging both commands were run and reproduced the saved
JSON exactly. Independent raw-logit rescoring and the frozen optimizer replay
audit are recorded in independent_audit.json. Raw tensors, checkpoints, training
data, full original source closure and internal project-recording helpers remain
at their original paths; do not upload the entire project root. Source metadata
retains absolute provenance and training-example indices. A complete portable
experiment release still requires separate packaging of permitted dependencies.

840 finite outputs, 240 comparisons, 48 exact repeats. Key-bias-only effects are
tiny under FP32 and nonzero under BF16. Restoration is not a general remedy.
Fixed batches/checkpoints and reused comparisons do not establish long-run
quality, statistical equivalence or generality. Publication remains on hold.
SOURCE_PROVENANCE.json and SHA256SUMS.json identify this local result capsule.
