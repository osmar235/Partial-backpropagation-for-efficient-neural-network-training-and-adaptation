# QKV confirmation independent review — 2026-10-26

**The prospectively defined directional confirmation passes in all four conditions.** Median session-level update-time savings are 5.23–7.14%. The original >=10% screening result remains failed and is not retroactively reclassified.

| Seed | Batch | Median saving | Conservative median interval | Positive sessions | Adjusted p |
|---|---:|---:|---|---:|---:|
| 99101 | 256 | 7.14% | [3.61, 8.63]% | 8/8 | 0.015625 |
| 99101 | 200 | 5.23% | [2.26, 7.60]% | 8/8 | 0.015625 |
| 99102 | 256 | 6.17% | [4.16, 9.79]% | 8/8 | 0.015625 |
| 99102 | 200 | 5.62% | [1.53, 8.09]% | 8/8 | 0.015625 |

Eight distinct process IDs and launch UUIDs, with no fragmented or replacement attempts. All 32 session-by-condition medians favor unbind; 123/128 individual pairs favor it. Five slower pairs are retained, all at batch 200 (worst -7.52%). The inferential unit is the session median, not individual pairs or updates. No discovery observations were pooled.
The one-sided exact sign test gives p=0.00390625 per condition and Bonferroni-adjusted p=0.015625. The frozen intervals use session extrema: marginal median coverage 99.21875%, with simultaneous coverage at least 96.875%, under independent identically distributed continuous session effects. Sequential sessions on shared hardware do not guarantee this assumption. The data support reproducibility on this setup; they are not independent task/hardware replications or proof of a universally fixed 5–7% gain.
Frozen read-only audit and a separate standard-library reconstruction agree on all pairings, session medians, statistics, initial/input identities, full loss sequences, final model/Adam/RNG hashes, mechanisms and counts. There are 256 trials, 33,024 measured updates, 5,120 discarded warmup updates, 320 numerical-check updates and 64 mechanism updates. No experiment was rerun during review.
All checked short-trajectory logits, losses, gradients, updated parameters and Adam moments match exactly (maximum checked difference zero). Every paired 129-update trajectory has identical recorded losses and final model/optimizer/RNG hashes, and the same condition repeats identically across sessions. This does not establish arbitrary-length numerical equivalence.
The expected graph change is present throughout: 19 native SelectBackward0 nodes become one plus six UnbindBackward0 nodes. Both arms retain math attention, FP32, strict determinism, full parameter gradients and the packed forward copy. Steady allocated peak and incremental memory match in every pair.
Order-stratified median session savings remain positive in all conditions, whether native or candidate runs first. These descriptive checks do not remove all possible timing confounds. Setup/first-update timings remain separate; no complete training saving is inferred.
All 24 idle-preflight samples show 1–7% utilization. Post-condition snapshots show GPU temperatures 61–70 C and SM clocks 2760–2782 MHz. These sparse snapshots are consistent with active benchmark work; they cannot certify absence of interference or thermal drift throughout every timed block.

| Seed | Batch | Native-first session median saving | Candidate-first session median saving |
|---|---:|---:|---:|
| 99101 | 256 | 7.53% | 6.86% |
| 99101 | 200 | 5.30% | 5.64% |
| 99102 | 256 | 6.12% | 6.75% |
| 99102 | 200 | 5.39% | 4.71% |

Decision: retain this as a confirmed modest operation-time improvement under the frozen configuration. Proceed next to separately specifying a matched real-data adaptation comparison of native versus unchanged unbind, including setup, data handling/augmentation, validation and checkpoint costs, with matched source states, data order and quality checks. Do not add another optimization before measuring transfer to actual training.
No real-data training, accuracy, complete-adaptation, broad hardware/model performance or novelty result is established. Publication remains on hold. No next experiment or new run command is prepared or launched by this review. All earlier evidence and the completed raw confirmation remain unchanged.
