# Stage 8 v2 validation record

2026-10-19. Project environment: `C:\Users\osmar\projects\.venv\Scripts\python.exe`.
All new checks used CPU/artificial inputs; no scientific GPU training was started.

## Historical v1 screen review

`scripts/review_stage8a_v1.py` passed for four real completed full-AdamW runs:
600 epochs, 105,600 updates. Verified recorded source hashes, configuration
fingerprints, exact LR schedules, finite losses/EMA, zero partial steps,
epoch/step timing and final metrics, checkpoint SHA-256 and AdamW counters.
All checkpoint optimizer states reached26,400 updates. The original failed
readiness decision is correct. Selected LR0.001: mean83.19%, minimum82.92%.
No calibration, real freeze or real evaluation output existed.

Outputs: `results/stage8A_review_v1/audit.json`, `per_seed.csv`, `REPORT.md`
and `preserved_v1_sha256.csv`. These are new review outputs, not raw-data edits.

## V2 mechanics and guard tests

`scripts/test_stage8_vit_v2.py`: **11 tests passed**.

- Native ViT partition and exact suffix gradients; no BatchNorm dependency.
- Exact suffix update and zero-gradient AdamW continuation with mixed labels.
- Replay prefix displacement and unchanged state on partial updates.
- Training-only45k/5k validation loader, with test-data construction forbidden.
- Five-epoch warm-up/cosine schedule and balanced ordering/nonadjacent masks.
- Matching completed-run reuse and changed-settings rejection.
- Evaluation loader cannot run without prior freeze.
- Poor baseline stops; both mean and minimum thresholds are independently enforced.
- Busy-GPU startup guard.
- MixUp image/label convex combinations, target normalization, smoothed soft-label
  CE and gradients equal the weighted hard-label formulation, independent paired
  streams reproduce whole batch sequences without advancing global dropout RNG.

## Complete artificial-data rehearsal

`scripts/stage8_workflow_v2.py smoke`: **PASS**, two-epoch tiny-model fake data,
two baseline plus two LGPB development runs, synthetic freeze, all24 evaluation
trajectories, all planned summaries/target reach counts, then integrity audit.
No CIFAR data or scientific seeds were used. Artificial readiness thresholds
are zero only inside the guarded synthetic namespace; real criteria stay85/83%.

`results/stage8_sanity_recipe_v2/evaluation/stage8_integrity_audit.json`:
PASS,24 runs/192 updates. Checked freeze/source provenance, paired initial
state, gates/EMA, LR, every MixUp lambda, periodic masks, timing aggregates,
checkpoint hashes and expected continuation/replay prefix counters.

Synthetic freeze SHA identifier:
`d38b7e64cdd999a86f5dea9c763166c65af006943d853304e379e437c60dbcd7`.
This is a synthetic diagnostic only; its accuracy/time values are not evidence.

The real v2 development/freeze/evaluation paths remain uncreated at preparation.
No GPU timing benefit, future baseline pass or architecture-transfer result is
claimed. See `preservation_check_v2.json` and `stage8_v2_preparation_sha256.csv`
for the final original-file preservation check and prepared-file manifest.
