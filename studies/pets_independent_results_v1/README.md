# Independent Pet evaluation results v1

Final 24-seed/72-trajectory evidence. Read REPORT.md and MANUSCRIPT_INSERT.md.
Frozen analysis and all primary/secondary outcomes are retained. This is a local
publication-support package; no external publication or upload has occurred.

From this package folder, reproduce final metrics, intervals and decisions:

    python scripts/reproduce_pets_evaluation.py --input evidence

Requires Python, NumPy and SciPy (versions recorded in the source freeze).
This portable analysis verifies the included summaries/logits against their
markers and reproduces the archived analysis without PyTorch, images or weights.
Evidence includes the run/source freeze, training gate/audit, 72 training records,
72 final-test logit sets, timing records, paired outcomes and completion hashes.

Large final checkpoints, cached image tensors, raw images and pretrained weights
are excluded. Original COMPLETE markers retain their hashes even when those files
are absent from the compact capsule. Full checkpoint/state/rescoring review requires
the original project layout, separate data/checkpoints and recorded CUDA runtime;
its entry point is scripts/review_pets_evaluation.py. The final review saved all
144 official-test rescoring checks, with zero maximum logit error.

Original preparation instructions are retained as source history, not a request
to repeat the evaluation. Internal project-history scripts stay in the main
project. SHA256SUMS.json seals actual capsule contents. All historical packages,
source files and original experimental results remain unchanged.
