# Stage8 v2 development review

Integrity review PASS: four runs,1200 epochs,211,200 updates. Baseline readiness PASS.
Source/configuration hashes, fixed split, gate/EMA, nonadjacency, LR, MixUp lambdas,
counts, timings, target crossings, checkpoint hashes/counters/finite states verified.
Saved checkpoints independently rescored on the training-derived validation split
using CPU. Exact fresh suffix gradients also checked at all four saved checkpoints.
No test images were loaded; no training was performed.

| Seed | Arm | Final validation | Partial fraction | Training seconds |
|---:|---|---:|---:|---:|
| 1001 | full_adamw | 85.46% | 0.00% | 1561.0 |
| 1001 | lgpb | 70.48% | 39.73% | 1355.4 |
| 1002 | full_adamw | 86.08% | 0.00% | 1484.5 |
| 1002 | lgpb | 72.40% | 39.78% | 1355.5 |

Full AdamW mean 85.77%; LGPB mean 71.44%; difference -14.33pp.
This is a large adverse development result. It must be retained and disclosed.
Two development seeds do not support a six-seed evaluation inference or a causal
claim that MixUp, loss selection, prefix size or AdamW continuation caused the gap.
Sequential baseline-then-LGPB ordering also limits interpretation of wall ratios.
See all learning curves and telemetry; no best epochs replace final outcomes.

Recommendation: keep the prospectively declared four-arm evaluation unchanged,
including unfavorable results, to characterize this architecture/recipe boundary.
Baseline readiness was the declared quality gate; no LGPB efficacy gate was declared.
Do not invent one retrospectively or tune a third recipe to erase this warning.
Reserved-seed evaluation will compare periodic and replay controls as planned.
This does not guarantee positive transfer; no implementation integrity defect
was detected by these checks, which cannot exclude every possible defect.

Freeze must follow this review and precede any evaluation. All new results remain
separate from Stage6 confirmation and the completed Stage7 manuscript evidence.
