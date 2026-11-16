# AGRLS-v0.3 diagnostic package

This version changes the experiment after analysis of the first d=20 ill-conditioned trace.

## Why v0.3 exists

The earlier AGRLS-v0 rebuilt a diagonal local surrogate using four forward probes at **every** accepted-move attempt. That made the surrogate very accurate, but it also meant:

1. forward-pass cost grew rapidly;
2. the experiment was not actually testing how long one gradient-anchored map remained valid;
3. the exact gradient mostly selected a fixed low-dimensional subspace, while fresh forward probes repeatedly rebuilt the local model inside that subspace.

AGRLS-v0.3 tests the intended idea more directly.

## v0.3 changes

At each exact-gradient refresh:

- compute the exact gradient once;
- build a 2-D subspace;
- use the exact projected gradient as the surrogate linear term;
- spend only **three** forward probes to estimate a full symmetric 2x2 quadratic curvature matrix, including the cross term;
- keep that surrogate and take candidate steps using only **one forward evaluation per candidate**;
- use each observed candidate loss to update the local curvature fit;
- request another exact gradient when the 2-D subspace is exhausted or repeated model disagreement occurs.

Thus, unlike v0.1, v0.3 does **not** spend four new mapping probes at every step.

## Run

From CMD, with your existing venv activated and this package copied to:

`C:\Users\osmar\projects\Projeto_AI_Fronteira`

run:

```cmd
python agrls_stage0_v0_3.py --objectives illcond_quadratic --dims 20 --seeds 0 --max-updates 60 --out stage0_v03_illcond_d20_seed0
```

Or double-click / run:

```cmd
run_v03_illcond_d20_seed0.cmd
```

## Upload after completion

Please upload the generated `stage0_v03_illcond_d20_seed0` folder, ideally zipped. The most important files are:

- `report.txt`
- `all_runs_summary.csv`
- `illcond_quadratic_d20_seed0\summary.csv`
- `illcond_quadratic_d20_seed0\trace_agrls_v0_3.csv`
- `illcond_quadratic_d20_seed0\agrls_v0_3_validity_horizon.png`
- `illcond_quadratic_d20_seed0\gap_vs_gradient_calls.png`
- `illcond_quadratic_d20_seed0\gap_vs_equiv_cost_r3.png`

Do not launch the full multi-seed experiment yet.
