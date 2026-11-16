# Stage8 frozen ViT evaluation: independent review

All24 runs /1,411,200 updates passed trace, provenance, checkpoint/counter and finite-state checks.
All five planned effects and all15 target contrasts were independently recomputed.
Frozen development inputs match their hashes. No training or dataset loading performed.

| Arm | Mean final test accuracy | Mean training wall | Mean partial use |
|---|---:|---:|---:|
| full_adamw | 86.402% | 1661.74s | 0.000% |
| lgpb | 71.697% | 1517.68s | 39.753% |
| periodic_matched | 85.175% | 1517.82s | 39.784% |
| replay_prefix | 72.058% | 1544.11s | 39.735% |

| Contrast | Accuracy delta pp [95% CI] | Training wall ratio [95% CI] |
|---|---:|---:|
| lgpb / full_adamw | -14.705 [-15.865, -13.545] | 0.9133 [0.9065, 0.9203] |
| periodic_matched / full_adamw | -1.227 [-1.740, -0.714] | 0.9134 [0.9062, 0.9207] |
| replay_prefix / full_adamw | -14.343 [-16.538, -12.149] | 0.9292 [0.9217, 0.9368] |
| lgpb / periodic_matched | -13.478 [-14.608, -12.349] | 0.9999 [0.9985, 1.0013] |
| lgpb / replay_prefix | -0.362 [-3.674, +2.950] | 0.9829 [0.9787, 0.9871] |

The new joint primary criterion FAILED: accuracy failed, speed passed. Original Stage6 inference is unchanged.
LGPB is less accurate in all six seeds. End-of-training wall savings are not useful time-to-quality savings:
LGPB reached70% in all seeds but required7.61x baseline training time (95% CI6.45--8.99);
it reached neither80% nor85% in any seed. Replay also reached neither; periodic reached80% in6/6
and85% at some epoch in5/6. Final-epoch and ever-reached accuracy are distinct.

Periodic is much more accurate at nearly identical partial use and training time. This whole-algorithm
contrast is adverse evidence for the loss-informed policy in this ViT/MixUp/partition recipe.
Replay does not remedy the large loss. These controls do not isolate a unique cause, prove all
transformers fail, or identify MixUp as causal. Mixed-batch gate associations are descriptive.
Mechanism/target intervals are unadjusted. Different recipes/datasets/hardware remain untested.
Retain all runs and telemetry; inspect timing drift rather than deleting outliers.

Decision: all planned scientific training is complete. No replacement seeds, third recipe or hidden
tuning. Revise the manuscript around workload-specific benefits and this explicit failure boundary.
Disclose the failed v1 screen, reused development validation and adverse v2 calibration. Complete
public code/data archiving and editorial packaging before submission; no public release is implied.
