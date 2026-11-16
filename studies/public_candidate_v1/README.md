# Paper-support code and evidence — local candidate v1

This is the single consolidated folder for the code currently selected to
support the evolving paper. It contains the original positive and negative
study sources, not only the latest favorable result. It is local preparation,
not a published release or a final manuscript supplement.

The latest result: replacing separate packed Q/K/V selections with unbind
reduces the measured complete five-epoch adaptation time by **6.34% / 6.40%**
across two fixed source checkpoints and eight paired sessions each. Recorded
learning outcomes match exactly. The result is specific to the tested
deterministic FP32/math-attention implementation. The original 10% discovery
gate remains failed. Unbind-based QKV splitting already exists in established
implementations; see the prior-art note below.

| Folder/file | Purpose |
|---|---|
| `qkv/src/qkv_unbind.py` | Byte-identical measured candidate, no legacy project imports |
| `qkv/tools/check_adapter.py` | Standalone small output/gradient/parameter-identity check |
| `qkv/tools/analyze_training.py` | Standard-library recomputation of all 16 audited timing pairs |
| `qkv/evidence/` | Audited pairs, component/quality tables and three QKV review reports |
| `archived_study_sources/` | Exact experiment, audit, test, launcher, specification and protocol source copies; original relative structure retained |
| `evidence/reviews/` | 27 study review/objective reports, including adverse results |
| `legacy_stage6b/` | Unmodified copy of the earlier Stage6B release candidate, with its original notices and licenses; historical scope |
| `SOURCE_PROVENANCE.json` | Original project-relative path and SHA-256 for each copied file |
| `SHA256SUMS.json` | Integrity manifest of this package, excluding itself |
| `verify_package.py` | Standard-library package integrity verification |

Start with [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md),
[REPRODUCTION.md](REPRODUCTION.md), and
[PRIOR_ART_AND_NEXT_STEPS.md](PRIOR_ART_AND_NEXT_STEPS.md).

From this folder, with Python on PATH:

```cmd
python -B verify_package.py
python -B qkv\tools\analyze_training.py
python -B qkv\tools\check_adapter.py --device cpu
```

The first two commands use only the standard library. The adapter check needs
PyTorch; add `--device cuda` for the optional GPU check. These commands do not
launch the original training studies or establish performance on another machine.

The original full workflows are preserved for inspection. They still require
their archived data, checkpoint and provenance inputs, and have not been made
portable by copying them. The exact remaining release work is recorded in
[RELEASE_STATUS.md](RELEASE_STATUS.md).
