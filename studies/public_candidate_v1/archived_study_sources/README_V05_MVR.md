# AGRLS-v0.5 — Marginal Value Refresh (MVR)

This package is the next Stage-0 development experiment.

## Scientific purpose

The previous seed-0 results showed that a variable gradient-refresh schedule can strictly dominate refreshing every step. Seed 0 is now formally designated as the development trajectory.

AGRLS-v0.5 replaces a hand-chosen update-number switch with a state-dependent marginal-value rule.

After a gradient refresh, let `G1` be the actual improvement from the first accepted move. Before paying for another forward-only trial, the surrogate predicts the next improvement `G_cont`.

For a synthetic gradient-to-forward cost ratio `r`, MVR refreshes when:

`G_cont / G1 < 1 / (r + 4)`

The `+4` represents three forward map probes plus the next trial forward needed to realize the benefit of a new gradient/subspace.

The run compares:

- fixed K=1
- fixed K=3
- phase 3->1 at update 30 (development reference only)
- MVR calibrated for r=2
- MVR calibrated for r=3
- MVR calibrated for r=4

## Run

From CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_v05_mvr_rosenbrock_d20_seed0.cmd
```

Or directly:

```cmd
python agrls_stage0_v0_5_marginal_value.py --objective rosenbrock --dim 20 --seed 0 --max-updates 120 --out stage0_v05_mvr_rosenbrock_d20_seed0
```

Upload the resulting folder as a ZIP.

## Important methodological rule

Do not interpret seed 0 as held-out confirmation. It has been used repeatedly to design the method and is now development-only. Once v0.5 is frozen, subsequent seeds will be used for validation.
