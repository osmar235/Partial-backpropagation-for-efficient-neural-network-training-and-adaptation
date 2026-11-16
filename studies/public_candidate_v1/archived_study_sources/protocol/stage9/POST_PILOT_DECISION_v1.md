# Stage9E decision — 2026-10-21

The prospectively frozen five-arm quality pilot is complete and failed its
practical objective. Both adaptive candidates failed both speed and final-
accuracy requirements on both development seeds. Full baselines reached the
primary sustained80% target; the result is not a readiness-gate stop.

Mean final validation accuracy: full81.72%, periodic81.40%, random81.15%,
boundary RMS80.19%, MixUp-only79.07%. Boundary took35.35% /35.47% longer
than the fastest blind control to target. Its fixed-update training-wall
saving15.7% did not translate into useful time-to-quality improvement.
All176,000 updates passed the frozen audit. Independent target/screen
calculations agreed; all ten saved endpoints reproduced their final
validation accuracy on rescore. No test evaluation or new training in review.

Limits disclosed: only two development seeds and reused validation; most
adaptive omissions were quota-forced; seeded full-warmup trajectories showed
numerical drift before omissions, despite matched initialization/RNG streams.
The exact cause was not isolated and strict deterministic execution was not
enforced. These facts prevent universal or tightly causal conclusions. They
do not satisfy or erase the failed development screen.

Stop this tested controller/cut4/40%-quota recipe. No automatic longer run,
lower omission rate, alternate cut, architecture switch, new seeds or
normalization search. Preserve sources, freeze, raw data and unfavorable
outcomes. Manuscript v7 remains a historical evidence record; publication
preparation stays on hold.

Return to design work before another GPU budget: define a distinct learning-
value/error-control mechanism and a reproducible comparison that could beat
efficient blind/cheap schedules. This decision does not claim that such a
mechanism has been found, or that all selective-gradient methods cannot work.
Any future study needs a new prospective justification, not a retrospective
relabeling of these same outcomes. Current master:v6.8.

Evidence: `results/stage9E_review_v1/REPORT.md`, `independent_audit.json`,
`checkpoint_validation_rescore.csv` and unchanged original
`results/stage9E_quality_pilot_v1/`. No next training command is recommended.
