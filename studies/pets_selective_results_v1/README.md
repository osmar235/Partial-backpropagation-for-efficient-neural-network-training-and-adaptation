# Pet selective development results v1

Read REPORT.md. This capsule adds compact development evidence and reproducible
analysis to the preserved source/protocol/preparation capsule. No external release
has occurred. Historical scientific packages and result files remain unchanged.

Recompute descriptive results and precision projections from this folder:

    python scripts/analyze_pets_selective_results.py --input evidence

Requires Python, NumPy and SciPy. Evidence includes frozen recipe, 18 final-logit
sets, summaries, steps, epochs, stream hashes and original audit. No images,
cached inputs, pretrained weights or final training checkpoints are redistributed.
COMPLETE markers retain original checkpoint hashes, although those files are
omitted. Full integrity review requires the original project layout and separate
data/checkpoints plus the recorded PyTorch/CUDA environment. Read-only reviewer:
scripts/review_pets_selective.py. Source execution remains as documented in
README_PETS_SELECTIVE.md; it is not an instruction to rerun completed experiments.

SHA256SUMS.json seals the actual capsule contents. Internal project-maintenance
scripts remain in the main project only. Planning recommends 24 independent
evaluation seeds; official-test access and final evaluation code are not enabled.
