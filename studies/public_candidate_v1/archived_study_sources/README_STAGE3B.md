# AGRLS Stage 3B — conservative gradient-horizon cap diagnostic

This is a **development-only seed-0** follow-up to Stage 3A.

Stage 3A showed the first positive end-to-end scaling signal on the heavier CNN:
- 23.33% fewer backward passes,
- 12.73% lower calibrated pure NN compute,
- 13.21% lower direct measured F+B time,
- 11.18% lower wall time,
- but a 1.36 percentage-point evaluation-accuracy deficit vs Adam.

Trace inspection showed one unusually long early cycle: a single exact gradient supported six accepted updates. The Stage 3B question is whether a **conservative information-horizon cap** can reduce trajectory drift while preserving most of the compute savings.

The only controller change in the candidate is:
- v0.10 baseline: `max_without_grad=6`
- v0.11 conservative candidate: `max_without_grad=2`

A cap of 2 means one Adam anchor plus at most one accepted surrogate continuation before another exact gradient is required. All other v0.10 logic is unchanged.

## Run

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage3b_v11_cnn_horizon_cap_seed0.cmd
```

or

```cmd
python agrls_stage3b_cnn_horizon_cap_v11.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --adam-lr 0.001 --out stage3b_fashionmnist_cnn_horizon_cap_seed0
```

The run compares Adam, v0.10/cap-6, and v0.11/cap-2 from the same initialization and dataset subset.
