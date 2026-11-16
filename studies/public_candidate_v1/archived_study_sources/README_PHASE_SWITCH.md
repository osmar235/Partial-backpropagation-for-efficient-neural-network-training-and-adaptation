# AGRLS-v0.4 phase-switch K diagnostic

This is the next controlled Stage-0 experiment.

## Why this experiment exists

The user-machine fixed-K sweep showed that no single refresh interval is best across the full Rosenbrock trajectory:

- K=3 is strongly cost-efficient during early/middle optimization.
- K=1 becomes preferable for deeper refinement.

This experiment tests whether **changing K during training** can outperform both fixed schedules.

This is not the final adaptive algorithm. The switch points are intentionally simple upper-bound/diagnostic policies.

## Policies

- `fixed1`: exact-gradient refresh after every accepted move.
- `fixed3`: refresh every 3 accepted moves.
- `phase3to1_25`: K=3 until accepted update 25, then K=1.
- `phase3to1_30`: K=3 until accepted update 30, then K=1.
- `phase3to2to1`: K=3 before update 40, K=2 for updates 40-79, then K=1.

The 25/30 switch points were selected from development-only diagnostics and must therefore be treated as hypothesis-generating, not unbiased benchmarks.

## Run

From Windows CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_v04_phase_switch_rosenbrock_d20_seed0.cmd
```

Equivalent direct command:

```cmd
python agrls_stage0_v0_4_phase_switch.py --objective rosenbrock --dim 20 --seed 0 --max-updates 120 --policies fixed1 fixed3 phase3to1_25 phase3to1_30 phase3to2to1 --out stage0_v04_phase_switch_rosenbrock_d20_seed0
```

## What to upload

Zip the complete folder:

`stage0_v04_phase_switch_rosenbrock_d20_seed0`

and upload it to ChatGPT.

## Interpretation rule

A phase-switch policy is interesting if it reaches a lower objective at lower equivalent cost than the relevant fixed policies. The main scientific goal is to establish that a **time-varying gradient-refresh frequency can occupy a better Pareto frontier** than a single fixed K.
