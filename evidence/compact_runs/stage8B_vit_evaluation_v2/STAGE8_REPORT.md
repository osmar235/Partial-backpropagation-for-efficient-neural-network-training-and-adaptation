# Stage 8 ViT four-arm results

New Stage 8 primary criteria; separate from Stage 6

| Variant | Reference | Accuracy delta pp [95% CI] | Wall ratio [95% CI] |
|---|---|---:|---:|
| lgpb | full_adamw | -14.705 [-15.865, -13.545] | 0.9133 [0.9065, 0.9203] |
| periodic_matched | full_adamw | -1.227 [-1.740, -0.714] | 0.9134 [0.9062, 0.9207] |
| replay_prefix | full_adamw | -14.343 [-16.538, -12.149] | 0.9292 [0.9217, 0.9368] |
| lgpb | periodic_matched | -13.478 [-14.608, -12.349] | 0.9999 [0.9985, 1.0013] |
| lgpb | replay_prefix | -0.362 [-3.674, +2.950] | 0.9829 [0.9787, 0.9871] |

Primary criteria apply only to the new frozen Stage 8 LGPB/full-AdamW contrast.
All mechanism/target intervals are descriptive and unadjusted. Inspect telemetry and all seeds.
Training-only wall excludes warm-up, validation, telemetry queries and final checkpoint writes.
Total experiment elapsed seconds separately include these per-run costs; initial data/model construction is excluded.
