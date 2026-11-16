# Stage14B decision — 2026-10-24

Integrity PASS. Both full references qualify. Both spatial and similarity fail
the frozen two-seed quality/resource screen. Decision: **park_fixed_recipe**.

Final full accuracy: 84.72% / 83.90%; similarity: 83.92% / 83.64%; spatial:
83.42% / 82.92%. Similarity gaps of 0.80 / 0.26 points pass the 0.30-point
quality rule only in the second seed. Full receives the same five-epoch
adaptation budget and remains the best control in both seeds.

Similarity exceeds its own starting accuracy by 1.76 / 1.50 points and exceeds
the original full source by 1.46 / 0.90 points. Those absolute gains do not
establish recovery relative to equally adapted full training. The gap widens
in seed 1501 and narrows in seed 1502. Preserve this distinction in discussion.

Charged recovery time falls 22.55% / 21.51%, with 26.39% lower maximum
incremental training allocation (18.48% lower absolute training peak). Adding
the original training cost leaves only 1.10% / 1.07% savings, still without
matched final quality in both seeds. No 22% total training-speedup claim.

No additional epochs, best-checkpoint substitution or threshold changes.
Return to distinct mechanism design; no new experiment is prepared here.
Retain all prior failures and this bounded result in paper memory v5. No
general refutation of merging, novelty claim or publication promotion.

Evidence: ../../results/stage14B_review_v1/REPORT.md. All 12 saved-model
scoring conditions reproduce logits exactly; 1,634 protected files unchanged.
