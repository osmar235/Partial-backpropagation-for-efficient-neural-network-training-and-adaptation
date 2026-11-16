# AGRLS-v0.8.1 seed-8 diagnostic

Purpose: diagnose and fix the frozen v0.8 seed-8 safety-limit failure without pretending that the patched run belongs to the original preregistered held-out batch.

Root cause: with measured r_hw < 2, retry_budget became 1. A rejected first Adam anchor caused an immediate gradient refresh at the same unchanged theta, resetting the full anchor step and potentially creating an endless reject-refresh cycle.

v0.8.1 fix:
- anchor rejections backtrack by halving the same step while reusing the same exact gradient;
- the cost-aware retry budget still applies to continuation-model misses after an accepted move;
- at most 12 anchor halvings are allowed;
- if Adam momentum is not a descent direction relative to the current exact gradient, the anchor direction falls back to steepest descent for that cycle.

Run `run_v081_seed8_diagnostic.cmd`.

Expected output folder: `stage1f_v081_seed8_diagnostic`.

Scientific status: seed 8 is now a development diagnostic for v0.8.1 because the fix was designed after observing its failure. If it passes, use fresh unseen seeds 10-18 for the next held-out validation.
