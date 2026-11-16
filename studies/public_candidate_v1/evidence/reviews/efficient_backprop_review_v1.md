# Efficient full-backpropagation backend review — 2026-10-25

The experiment completed. The terminal exception was a review CLI path bug: the frozen auditor compared an absolute saved attempt parent to a relative output path. It did not indicate incomplete training or corrupt results.
The new read-only wrapper resolves its input path, verifies the root completion marker, and calls the unchanged frozen auditor. Relative/absolute regression calls produce identical PASS results and leave all raw files unchanged. Do not rerun the timing screen to fix the CLI.

| Seed | Batch | Math ms | Efficient ms | Paired slowdown | Decision |
|---|---:|---:|---:|---:|---|
| 99071 | 256 | 31.574 | 34.714 | 10.46% | FAIL_COST |
| 99071 | 200 | 23.889 | 26.465 | 12.35% | FAIL_COST |
| 99072 | 256 | 29.886 | — | — | FAIL_NUMERICAL |
| 99072 | 200 | 23.473 | 26.251 | 11.67% | FAIL_COST |

No alternative qualifies under the frozen FP32/strict-determinism rule. Efficient attention is slower by median paired 10.46%, 12.35% and 11.67% in its three numerically passing conditions. Only one of 18 paired rounds meets the 10% saving target; none of the three condition medians does. Keep all timing variability, including the slow first math trial, rather than deleting rounds retrospectively.
The fourth efficient condition (99072/batch 256) fails the second-update parameter check for block zero's packed attention projection bias: maximum absolute error 3.12618e-5, scaled error 1.56308. Its logits, gradients, moments, RNG match and repeatability checks pass. Timing is correctly absent. Unlike the earlier small diagnostic, these saved full-shape metrics do not isolate which query/key/value slice differs; do not claim a new full-shape key-bias localization.
Flash is unavailable in this torch build. cuDNN attention is unsupported under the preserved strict-determinism setting. These are compatibility outcomes, not measured speed failures. Do not generalize this result to mixed precision, other builds/hardware, relaxed determinism or longer sequences.
Efficient attention reduces allocated memory in the three timed conditions. The paired memory ratios are retained in independent_comparisons.csv; memory savings do not substitute for the primary time gate. The benchmark measures artificial steady operation cost, not complete adaptation, real quality or novel learning behavior.
Verified 42 timing trials, 378 timed updates, 126 discarded warmups, 40 correctness updates and 16 capability attempts. The latter include unsupported probes which can stop partway. Timing repeats are not independent training replications.

## Decision

Close this fixed backend-only screen without promoting an alternative. Preserve the original math baseline for current comparisons. Do not loosen tolerances, enable reduced precision or relax determinism retrospectively. A targeted state-compatible attention/QKV layout change remains a separate possible next study, requiring its own prospective correctness and cost checks; it has not been prepared or launched here.
Publication remains on hold. All prior/raw evidence and frozen sources are preserved. Master v7.28 and paper memory v17 are current.

## Review-only command

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\review_efficient_backprop_completed.cmd
```

The old experiment CMD is preserved as frozen evidence; its final relative-path audit invocation remains historical. Use this new review-only command if rechecking results. The assistant has already run both relative and absolute review paths successfully.
