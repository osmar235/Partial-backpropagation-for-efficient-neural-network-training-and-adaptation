# Reviewed BF16 matched adaptation results v1

Read REPORT.md for complete results, scope and limitations. Both BF16 policies
pass the predeclared practical screen on two checkpoints; automatic saves about
43% of marginal adaptation time on this RTX 5080 setup. This establishes a
conventional execution baseline, not a new gradient method or universal result.
Earlier gates remain failed. Repeated sessions are not extra independent seeds.

Reproduce grouped statistics with standard Python, from this folder:

```cmd
python -B analysis\summarize_results_v1.py review_data.json --output reproduced_groups.json
```

Use a fresh output path; compare with group_summary.json. This command was run
during packaging and reproduced the JSON exactly. It needs no GPU or torch.
review_data.json includes all 36 trajectory summaries and 24 paired comparisons;
validation_scores.json contains all 252 independent saved-logit scores.

Frozen experiment code/protocol remain in ../bf16_continuation_addendum_v1.
Raw checkpoints/logits, CIFAR data and internal review/recording helpers remain
at original project paths and are not copied into this results capsule. The
source capsule retains local absolute provenance; a full portable experiment
release still requires separate dependency packaging. No external publication
or release has occurred. SOURCE_PROVENANCE.json and SHA256SUMS.json identify
this local collection; all earlier evidence packages remain unchanged.
