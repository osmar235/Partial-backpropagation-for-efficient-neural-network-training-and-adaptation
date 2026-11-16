# Stage 7E AdamW robustness — independent review

Reviewed 2026-10-18. Six CIFAR-10 seeds 69–74, full-gradient AdamW versus
unchanged LGPB gate/partition with AdamW; all 12 runs complete.

## Finding

LGPB reduced training wall time by **9.66%** with a mean accuracy difference
of **+0.028 pp**. The exploratory accuracy interval includes zero: there is no
clear accuracy advantage or loss in these runs. This supports efficiency under
this additional fixed AdamW/weight-decay recipe, without establishing formal
equivalence, new non-inferiority, or an optimized modern training recipe.

## Audit and provenance

All 70,560 step records and 360 epochs passed checks of settings, source hashes,
fingerprints, alternating run order, loss EMA/gate decisions, cosine rates,
mode counts, component/timing reconciliation, final metrics and all target
crossings. All reported paired effects and confidence intervals reproduced.
All 218 recorded original Stage 6/release files remain unchanged.

The configuration specifies AdamW, decoupled weight decay 0.0005, LR 0.0003,
30 epochs, batch 256, EMA 0.95, original prefix and 196 excluded warm-up steps.
The run label `adam` denotes **full-gradient AdamW**, not ordinary Adam here.
LGPB zero prefix gradients allow AdamW's moment/step updates and decoupled
prefix weight decay to continue on partial steps, as implemented in the frozen
Stage 7 harness. No optimizer/controller setting was selected from these results.

The input hash manifest is in `audit.json`; the audit writes only this separate
review folder. Raw traces cannot independently verify learned parameters or
optimizer state because no checkpoints were supplied. Warm-up behavior is
supported by source/configuration rather than a separate warm-up trace.

## Results

| Quantity | Full-gradient AdamW | LGPB with AdamW |
|---|---:|---:|
| Mean final accuracy | 84.955% | 84.983% |
| Mean training wall time | 394.00 s | 355.92 s |
| Mean partial-update fraction | 0% | 35.774% |
| Mean full-step backward region | 39.033 ms | 38.858 ms |
| Mean partial-step backward region | — | 22.269 ms |
| Mean wall-minus-component overhead | 14.43 s | 13.81 s |

Paired final accuracy, LGPB minus full-gradient AdamW: **+0.028 pp**,
exploratory paired 95% t interval **-0.106 to +0.163 pp**. Four seeds had higher
accuracy with LGPB and two had lower accuracy; deltas ranged from -0.17 to
+0.16 pp. This is not proof of equal accuracy or superiority.

Geometric paired training-wall ratio, LGPB/full-gradient AdamW: **0.9034**
(95% interval **0.8930–0.9139**), or **9.66% less wall time**
(transformed interval **8.61%–10.70%**). Every seed was faster with LGPB;
individual reductions were 8.91%–11.58%. The geometric paired reduction differs
slightly from a reduction computed from arithmetic mean times.

All intervals use six pairs and five degrees of freedom, are exploratory and
unadjusted for multiple comparisons. Stage 6B inferential gates are not reused.
Mode timers include gradient assignment/zero construction and synchronization
conventions; they are not isolated GPU-kernel or theoretical-FLOP measurements.

## All predeclared time-to-quality targets

Both variants reached every target in all six seeds. Times are first observed
epoch-evaluation crossings; there is no within-epoch interpolation.

| Accuracy target | Geometric LGPB/AdamW time ratio | Exploratory 95% CI |
|---|---:|---:|
| 60% | 1.0064 | 0.8865–1.1425 |
| 70% | 0.9042 | 0.8265–0.9892 |
| 80% | 0.9705 | 0.9001–1.0464 |

The 80% target is numerically 2.95% sooner but its interval includes one;
do not describe training-wall savings as automatically equal time-to-quality
savings. Report all targets rather than choosing the favorable 70% crossing.

## Scope and next action

Stage 7A–7E are complete, including all 96 substantive training runs. Seeds
69–74 are now seen. The controller, recipe, original Stage 6B evidence and
all prior negative/inconclusive results remain unchanged.

This AdamW result addresses the zero-weight-decay recipe limitation on the
same CIFAR-10/ResNet-GN workload. It does not establish another architecture,
hardware platform, tuned augmentation recipe, loss-gate superiority, or removal
of the CIFAR-100 accuracy cost. Do not pool absolute timings or compare Adam
and AdamW across different phase/seed blocks as an optimizer-family experiment.

The next task is manuscript consolidation, not another analysis run. Use
`protocol/stage7/FINAL_REVIEW_SYNTHESIS.md` for the evidence and revision map.
No manuscript or public-release artifact was edited or published in this audit.

Reproduce without training:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
"C:\Users\osmar\projects\.venv\Scripts\python.exe" -B scripts\analyze_stage7e_review.py
```
