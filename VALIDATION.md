# Publication-package validation

Checked on 15 November 2026 with Python 3.14.3, NumPy 2.4.3 and SciPy 1.17.1.

All nine replay steps in `tools/reproduce_analyses.py` passed: original CNN confirmation; QKV adaptation; competitive attention reference; BF16 continuation; optimized prefix cost; prefix adaptation; Pet baseline; Pet selective development; independent Pet evaluation. The CNN confirmation and three grouped ViT exports matched their archived JSON summaries exactly. The independent Pet replay verified included completion-marker hashes, reconstructed official-test scores from logits and matched the archived frozen analysis.

The repository hash verifier passed. No training experiment was rerun. The included source copies retain their recorded hashes; the older collection omits three internal documents, as recorded in `SOURCE_PROVENANCE.json` and its publication-subset notice. Original local archives remain unchanged.

This validation covers the distributed evidence and selected analysis calculations. It does not certify every historical launcher in a clean GPU environment or regenerate excluded model checkpoints.
