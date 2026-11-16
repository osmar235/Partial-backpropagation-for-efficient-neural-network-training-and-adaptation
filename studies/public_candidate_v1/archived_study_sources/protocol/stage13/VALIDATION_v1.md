# Stage13A preparation validation — 2026-10-22

Five CPU and five CUDA unit tests passed:

- Local gradients equal separately constructed per-block objectives; losses do
  not cross the detached boundaries.
- Final loss and final-block gradients agree across all modes. Connected
  auxiliary supervision gives a detectably different early gradient from local
  objectives, and suffix-only gradients are absent in the prefix.
- Restored zero moments reproduce fresh Adam updates exactly. Inactive weights,
  moments and BN buffers obey the declared semantics in all four modes.
- Calibration/probe/score pools are mutually disjoint and index streams replay.
- Native pretrained ResNet executes local updates with finite gradients/states
  and changes early-block parameters while preserving final-block gradient.

Additional preparation-only native CUDA check at batch128 /96px passed the
auditor's aggregate tolerances. All four final-block gradients had identical
double-accumulated squared norm and reference dot product33.281082609753454.
This used artificial inputs, not the real probe pools. Code:
protocol/stage13/native_gradient_check.py.

Complete artificial CPU/CUDA diagnostics passed: two source states, four modes,
32measured operations,8auxiliary calibration steps and8warmup operations each.
Audits verified raw hashes, disjoint index streams, calibrated backbone identity,
all first-trial saved states/counters, loss/gradient arithmetic and summary flags.
Runtime checks inspect finite gradients, frozen buffers and active/inactive
optimizer semantics on every measured update. Synthetic scientific flags are
irrelevant to real feasibility; both artificial runs flagged false.

Completed CPU resume re-audited without output writes. All before/after file
hashes are checked by the preparation verifier. Artificial outputs:
results/stage13_sanity_local_v1_final_cpu and
results/stage13_sanity_local_v1_final_gpu.

Prospective freeze:
c988ee42b72ed319c6df9c9f92a506ce69e80530b4023c6eb5b469bad15fd54e.
Exact preserved Stage12 head-warmup checkpoint hashes, pretrained input,
scientific dependencies, protocol/spec/tests/launcher and evidence are bound.
1,088prior immutable files verified. Only CURRENT_RELEASE.md and README_STAGE9.md
are intentionally updated existing pointers; master7.6 is new. Prior source
files, raw experiments, masters and manuscript remain intact.

No real Stage13A diagnostic was executed during preparation. Verify real output
absence at finish. This preparation establishes mechanics, not useful local
gradients on real data, convergence, accuracy retention or end-to-end speed.
CPU/GPU test results do not substitute for the user's real diagnostic.
