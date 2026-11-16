# Evidence ledger — 2026-10-26

The current QKV result and the earlier partial-gradient program are separate
claims. Findings below retain their original task, quality margin and cost
scope. Percentages from different clocks or final accuracies are not a direct
algorithm ranking. Source files are in `archived_study_sources/`, except the
Stage6B historical release in `legacy_stage6b/`.

| Line of work | Observed result / decision | Evidence and code location |
|---|---|---|
| Early probes, through Stage5 | Exploratory mechanism development; retain for history, not independent confirmation | Internal history catalog; selected dependencies and legacy release retain source provenance |
| Stage6B LGPB-MC | Original confirmatory result retained: about 10.24% time reduction, -0.548 pp accuracy versus Adam under its declared 1 pp noninferiority margin | `legacy_stage6b/`; original protocols and analysis remain authoritative |
| Stage7B/C/D/E | Gate superiority not established against random/periodic schedules; CIFAR-100 saves 9.98% with -0.695 pp; AdamW extension saves 9.66% with +0.028 pp mean difference, not formal equivalence | [7B](evidence/reviews/stage7B_review_v1.md), [7C](evidence/reviews/stage7C_review_v1.md), [7D](evidence/reviews/stage7D_review_v1.md), [7E](evidence/reviews/stage7E_review_v1.md); `scripts/stage7*`, `scripts/analyze_stage7*` |
| Stage8 ViT extension | LGPB 71.697% versus full 86.402%; approximately 8.67% wall saving with -14.705 pp, joint criterion fails. Periodic control much stronger | [8B](evidence/reviews/stage8B_review_v2.md); `scripts/stage8*`, `protocol/stage8/` |
| Stage9 boundary/control work | Cost diagnostics alone insufficient; both adaptive Stage9E recipes fail both seeds against strong blind schedules | [9C](evidence/reviews/stage9C_review_v1.md), [9D](evidence/reviews/stage9D_review_v1.md), [9E](evidence/reviews/stage9E_review_v1.md); `scripts/stage9*` |
| Stage10 representation / Haar | Full mean 82.15%, narrow 81.43%, coarse 74.96%, pixel 74.73%, Haar 74.84%; tested Haar allocation fails. Does not reject frequency methods in general | [10A](evidence/reviews/stage10A_review_v1.md); `scripts/stage10*` |
| Idea 1: Stage11 growth | Feasibility passes; fixed-run savings 23.54/21.99%, deficits 1.40/0.12 pp versus 0.20 pp limit; both-seed rule fails. Sustained 80% reached later | [11B](evidence/reviews/stage11B_review_v1.md); `scripts/stage11*` |
| Idea 2: Stage12 pretrained selective adaptation | Periodic saves 39.91/38.81%, deficits 1.52/0.96 pp; fails. Ordinary head-then-full control saves 11.66/11.38% with deficits 0.26/0.20 pp, within 0.30 pp margin | [12A](evidence/reviews/stage12A_review_v1.md); `scripts/stage12*` |
| Idea 3: Stage13 local objectives | Initial full reference failed; repaired Stage13B baseline qualifies. Stage13C local recipe still loses 8.26/12.18 pp and is 8.28/5.69% slower including calibration | [13C](evidence/reviews/stage13C_stage14A_review_v1.md); `scripts/stage13*` |
| Idea 4: Stage14 token compression | Similarity recovery saves 22.55/21.51% marginal recovery time but loses 0.80/0.26 pp; both-seed 0.30 pp rule fails. Source-inclusive savings 1.095/1.074% retained as a quality/cost tradeoff | [14B](evidence/reviews/stage14B_review_v1.md); `scripts/stage14*` |
| Stage14C optimizer-history follow-up | Reset gains 0.06/0.12 pp; remaining deficits 0.74/0.14 pp. Small directional finding retained; practical recipe still fails | [14C](evidence/reviews/stage14C_review_v1.md); `scripts/stage14_optimizer_history.py` |
| Stage15 side adaptation | Saves 20.71–26.19% but deficits 1.20/2.12 pp against strongest controls; fixed recipe closed | [15A](evidence/reviews/stage15A_review_v1.md); `scripts/stage15*` |
| Sampled weight gradients / profile | Tested implementation 4.97–13.82% slower than native; memory reduction does not replace time benefit. Native MLP weight-gradient device attribution only 7.66–8.56% | [Cost](evidence/reviews/sampled_weight_cost_review_v1.md), [profile](evidence/reviews/runtime_profile_review_v1.md); `scripts/sampled*`, `scripts/runtime*` |
| Efficient backend screen | Under strict FP32 policy, efficient SDPA is slower in three numerically passing conditions; fourth fails numerical gate. Flash/cuDNN unavailable or unsupported under that build/policy | [Backend](evidence/reviews/efficient_backprop_review_v1.md); `scripts/efficient_backprop.py` |
| QKV discovery | 5.21–7.14% savings; 23/24 pairs faster; original >=10% gate fails and remains failed | [Discovery](qkv/evidence/qkv_layout_review.md); `scripts/qkv_layout.py` |
| QKV operation confirmation | 5.23–7.14% across four conditions; all 32 session-condition medians positive, 123/128 pairs faster, checked learning records identical | [Confirmation](qkv/evidence/qkv_confirmation_review.md); `scripts/qkv_confirmation.py` |
| QKV real-data confirmation | 6.3377/6.4034% median complete adaptation-time reduction; 16/16 pairs faster; final 84.70/83.88%, exact paired recorded learning outcomes; no absolute peak-memory gain | [Training](qkv/evidence/qkv_training_review.md); `scripts/qkv_training.py`, portable core in `qkv/src/` |

## Claim boundaries for the paper

Eight hardware sessions repeat two source trajectories; they are not sixteen
independent model seeds. Confirmation does not establish architecture, dataset,
precision or hardware generality. Training-plus-augmentation also improves
(6.97/7.24%), so the observed benefit is not solely file loading/serialization.
The source pretraining cost is excluded from this adaptation claim.

The 1.1% token result should remain in discussion with its quality tradeoff and
accounting boundary. Local-learning and side-network failures apply to our
fixed recipes, not to every published method. Negative controls must remain
available wherever their results support a manuscript argument.

All four original ideas have received initial tests. See the preserved
[objective comparison](evidence/reviews/research_objective_comparison_v1.md)
and the separate proposed next-step queue. A new research question requires
a new protocol; original thresholds and exclusions stay fixed.
