# Stage11B preparation validation — 2026-10-22

No real Stage11B trajectory was executed during preparation.

- Five CPU and five CUDA tests passed: all five transition controls with
  nonzero learned classifier states, unchanged/mapped weights, fresh/continued
  moments, preserved LR, subsequent Adam counters, finite learning; primary
  screen inclusion of competitive small/reset controls, readiness and exact
  quality/time decision boundaries.
- Complete artificial CPU and CUDA workflows passed: ten runs and240 audited
  updates each, all midpoint/final states, exact growth mapping, data RNGs,
  bitwise-identical histories within matched architecture groups before growth,
  continued global LR/MixUp/dropout schedules and optimizer ages.
- Artificial results are not accuracy, runtime or feasibility evidence. CPU and
  GPU trajectories need not match each other; matched controls must agree within
  their own environment before the transition.
- Re-ran completed artificial CPU study with no output writes; the preparation
  verifier compares every saved file to its before-resume hash manifest.
- Stage11A's real read-only audit and independent post-review arithmetic passed.
  The latter verifies809 prior immutable files. Sealing the review/helper adds
  seven, for816 protected immutable inputs to this preparation.
- All32 scientific source/spec/protocol/test/launcher dependencies and four
  evidence files are bound to the prospective freeze. Real output must remain
  absent when preparation verification finishes.

Freeze hash:
`1cb04236fba56f0929b8d0fbbde523b5825993d5424a9abea89153fcb97ea6bc`.

Artificial outputs:
results/stage11_sanity_growth_pilot_v1_final_cpu and
results/stage11_sanity_growth_pilot_v1_final_gpu.

Preparation proof: pilot_preparation_preservation_v1.json,
pilot_cpu_before_resume_sha256_v1.json, pilot_preparation_verification_v1.json
and pilot_preparation_sha256_v1.json. Only CURRENT_RELEASE.md and README_STAGE9.md
are intentionally updated existing pointers; before/after hashes are recorded.
The old master7.1, README_STAGE11.md, prior results and manuscript remain intact.

Limits: tiny synthetic rehearsals cannot demonstrate real-data convergence or
hardware speed gains. Stage11A tested native width expansion and trainability;
Stage11B's efficacy is still unknown. Strict FP32/math-attention settings and
instrumentation costs limit any subsequent claim until broader controls exist.
