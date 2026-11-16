## Current action — independent Pet evaluation frozen — 2026-10-30

Ready for user-run evaluation: 24 fresh seeds 2801–2824 × full/frozen/periodic-r4
BF16 arms, 72 trajectories. Keep the qualified 2944/736 trainval split, 15 epochs,
690 updates, original LR/augmentation recipe, prefix stem+layer1+layer2 and
173 periodic refreshes. Repeat all six arm orders four times. Historical sources,
results and development decisions remain unchanged.

All training and its integrity audit must finish before official test.txt/images
are accessed. Then score final checkpoints on the official 3669-image test set in
common FP32 and BF16. Only complete, audited test results are summarized. No test
tuning, checkpoint selection, cadence sweep or quality-based early stopping.
Periodic-versus-full remains primary, freezing-versus-full secondary.

For each method's combined quality-and-time claim, require the one-sided 97.5%
lower paired-t accuracy bound above -0.30 pp AND the one-sided 97.5% upper paired-t
log complete-time-ratio bound below zero. Intersection-union alpha .025 within
each claim; Bonferroni across two claims controls family error at .05 under the
assumptions. No accuracy superiority claim. Own-policy scoring is robustness
analysis; uncertainty is conditional on this test sample/backbone/session.
Development SD planning projected half-width 0.271 pp (point SD) or 0.566 pp
(upper SD) at 24: do not promise the original 0.20 pp target or widen the margin.

Primary timing retains the prespecified adaptation-complete clock with internal
validation. Official-test scoring is a separate post-training assessment, timed
and disclosed separately, not silently included in the claimed adaptation saving.
One-time preparation/pretraining/imports/audits remain excluded. Fragmented
training suppresses automatic combined claims and requires review. Restarts
verify and skip completed training/scoring without choosing favorable attempts.

Preparation passed 12 functional checks across CPU/CUDA, including access gating,
artificial test-annotation/image parsing, cache tampering, train/test-id exclusion,
endpoint intersection and uncertainty/multiplicity thresholds. Full artificial
CPU and native-ResNet CUDA workflows passed: 12 synthetic training trajectories,
12 synthetic final scorings, 48 training-checkpoint rescores across initial and
restart audits, and two byte-identical complete restarts. Real source/data/weight/
runtime preflight passed; no real evaluation training or official-test access
occurred during preparation. Native smoke uses batch four and 224px; real batch
64 capacity was already established by baseline and development.

Run scripts/run_pets_evaluation.cmd; instructions in README_PETS_EVALUATION.md.
Send results/pets_independent_evaluation_v1 after COMPLETE. Allow approximately
45–60 minutes and roughly 15 GB free space; no additional download is required.
New scientific source/protocol/evidence capsule:
paper_support/pets_independent_evaluation_v1 (and ZIP). Internal maintenance
scripts remain separate. No publication or external release has occurred.
Current master v7.60 / memory v49. Manuscript v10 unchanged pending final results.
