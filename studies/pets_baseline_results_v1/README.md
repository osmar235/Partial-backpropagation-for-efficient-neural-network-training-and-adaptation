# Oxford Pet baseline results v1

Read REPORT.md. This is a local, publication-support capsule; nothing has been
published externally. Historical source/results packages remain unchanged.

Compact evidence contains development summaries, logits, epochs, step logs,
stream hashes, freeze and original audit. No images, cached inputs, pretrained
weights or final checkpoints are redistributed. COMPLETE.json retains the
original checkpoint hashes, although those large checkpoint files are omitted.

Recompute the descriptive table from this folder with Python and NumPy:

    python source/scripts/analyze_pets_baseline_results.py --input evidence

Full integrity review requires the separately held data and checkpoints in the
original project layout, plus torch/torchvision. Its entry point is
source/scripts/review_pets_baseline.py; use a fresh --output directory. It reads
saved evidence without training. The internal project-record updater is not part
of this public-support capsule. SHA256SUMS.json seals the capsule's actual files.
See the original frozen runtime in evidence/baseline_freeze.json.
