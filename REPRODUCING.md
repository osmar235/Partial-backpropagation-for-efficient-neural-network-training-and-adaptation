# Reproducing the analyses

The root replay runs the following checks in isolated output directories. Archived results are not overwritten.

| Replay | Included inputs | What it checks |
| --- | --- | --- |
| Original CNN confirmation | Nine-seed results in the historical Stage6B capsule | Accuracy and timing intervals and prespecified decisions |
| QKV adaptation | Paired timing records | Per-source timing summaries and direction tests |
| Competitive attention reference | 288 exported trials | Twelve grouped timing comparisons |
| BF16 continuation | Trajectory summaries and paired comparisons | Exact agreement with archived grouped statistics |
| Optimized prefix cost | Exported timing/component records | Exact agreement with archived grouped statistics |
| Prefix adaptation | Trajectory summaries and paired comparisons | Exact agreement with archived grouped statistics and accuracy screens |
| Pet baseline and development | Compact summaries and saved logits | Descriptive development analyses |
| Independent Pet evaluation | 72 training records and official-test logits | Marker hashes, scores, paired intervals and the frozen primary/secondary decisions, compared with the archived analysis |

The root verifier covers all distributed files. A replay output is a calculation from existing evidence, not a fresh GPU replication. CNN control and negative-result tables that do not have a root replay are available under `evidence/` and retain their original study-specific analyses in the source collection.

## Training and checkpoint audits

Frozen experiment code, specifications, seed lists, recorded environments and run markers are retained in the study capsules. Consult the README for the particular study. Training requires its recorded PyTorch/torchvision/CUDA environment, datasets and, for adaptation, the specified source checkpoints. The root NumPy/SciPy requirements are only for analysis.

Raw image datasets, tensor caches, pretrained weights and final model checkpoints are not bundled. Original completion markers can therefore contain hashes for files outside this compact distribution. The root manifest enumerates what is actually included. Pet final-test logits permit score reconstruction without redistributing the images; they do not permit rebuilding model states. For ViT adaptation, the compact grouped-statistic exports do not replace the source checkpoints or raw logit archive.

The historical training launchers have not all been certified in a clean checkout. Some retain original project-relative or machine paths and prerequisite-study references. A complete end-to-end replication requires resolving those paths and supplying the recorded prerequisites without changing protocols or thresholds. Do not interpret the portable analysis checks as certification of those launchers.

## Frozen results and timing scope

Keep source protocols and decision margins unchanged. The nine-seed CNN confirmation, two-source ViT adaptation and 24-seed Pet adaptation have different independent units and accuracy screens. Repeated timing sessions do not increase the number of independently trained source checkpoints. Hardware-specific timings should be measured again on the target system.

Historical instructions to launch or tune experiments are retained for provenance; they are not instructions to reopen the completed official-test evaluation. New exploratory work should use separate protocols and outputs.
