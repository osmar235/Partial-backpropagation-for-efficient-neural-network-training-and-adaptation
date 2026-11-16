# QKV unbind independent review — 2026-10-25

The fixed QKV-unbind implementation produces a modest, consistent artificial operation-time improvement, but FAILS the prospectively fixed >=10% saving rule in all four condition medians. Preserve both facts; do not relabel the cost gate as passed or discard the measured benefit.

| Seed | Batch | Native ms | QKV unbind ms | Median paired saving | Faster rounds / 6 | >=10% rounds / 6 |
|---|---:|---:|---:|---:|---:|---:|
| 99081 | 256 | 28.943 | 26.943 | 6.62% | 6 | 1 |
| 99081 | 200 | 22.336 | 21.081 | 5.36% | 5 | 0 |
| 99082 | 256 | 28.849 | 26.787 | 7.14% | 6 | 1 |
| 99082 | 200 | 22.461 | 21.234 | 5.21% | 6 | 0 |

Candidate time is lower in 23/24 paired rounds. One seed99081/batch200 pair is 0.34% slower. Exactly two individual pairs exceed 10% saving; no condition median does. Medians of paired ratios are the decision statistic, not ratios of median times. All rounds are retained. These are timing repetitions within a short screen, not 24 independent training studies or evidence of statistical significance across tasks/hardware.
All full-shape numerical checks pass with maximum checked absolute difference zero: logits, losses, parameter gradients, updated parameters and AdamW moments at the two checked updates. CPU/CUDA RNG and repeatability match. Across all 24 timed pairs, the nine-update loss sequences, final parameter hashes, final optimizer hashes and RNG hashes also match exactly. This is checked identity on the tested trajectories, not a proof for arbitrary training or other models.
The intended mechanism is present in every full-shape condition: native has 19 SelectBackward0 nodes; candidate has one plus six UnbindBackward0 nodes. Both still execute six math SDPA calls and 18 forward selects internally; unbind changes their autograd graph. There is no forward-copy removal or backend/precision/gradient-policy change. Full parameter gradients remain finite.
Steady-update peak and incremental allocated memory are identical within every pair. Setup and first-update timings are retained separately in the independent comparisons and raw trials; no complete-adaptation or cold-start saving is inferred from steady timing.
Verified 48 trials, 432 timed updates, 144 discarded warmups, 40 correctness updates and eight mechanism updates. GPU idle preflight and before/after telemetry were retained. Snapshot telemetry is not continuous monitoring. The frozen read-only auditor passes, and an independent reconstruction reproduces pairings, medians, identities, numerical results and mechanism counts. No experiment or training was rerun during review.

## Decision and next-step boundary

Close this fixed implementation as a failed >=10% feasibility screen. Retain it as a verified engineering improvement of approximately 5–7% on this workload and as a possible explicit control in a separately justified future experiment. Do not promote it as a qualifying method, automatically replace historical references, lower the threshold or launch real-data validation from this result.
The remaining packed QKV contiguous copy is unchanged. A separate hypothesis could target that repacking cost, comparing any redesign against both historical math and this unbind implementation so that the already-observed gain is not counted again as a new contribution. Such a rewrite can move copies elsewhere or alter numerical behavior; it requires a new prospective design, correctness checks and cost gate. It has not been prepared or run here.
This evidence does not establish novelty, real-data accuracy or generalization to optimized attention backends, larger shapes or other devices. Keep the exact-match finding, the graph mechanism, the modest time reduction and the failed practical threshold in the paper memory. Publication remains on hold.

Master v7.30 and paper memory v19 are current. All earlier/raw evidence and frozen sources were hash-verified and preserved.
