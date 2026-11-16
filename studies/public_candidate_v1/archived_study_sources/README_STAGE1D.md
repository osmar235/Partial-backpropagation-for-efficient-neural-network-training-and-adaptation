# AGRLS Stage 1D — Information-Matched 1-D Probe-Free MVR

## Purpose
Stage 1C showed that Adam anchoring restored near-Adam accuracy but the 2-D probe-free surrogate produced 149 model-miss rejections and 210 forward-only queries. The key diagnosis is structural: with the exact gradient known, a 2-D quadratic still has three unknown curvature coefficients, while the first mandatory forward trial supplies only one scalar curvature observation.

AGRLS-v0.8 therefore matches surrogate complexity to available information. It reuses a 1-D quadratic along the Adam-preconditioned direction. The exact gradient supplies the linear term, and the first accepted forward trial identifies the single curvature coefficient.

## New elements in v0.8
- Adam-preconditioned gradient anchor retained.
- Zero dedicated map probes.
- Reusable surrogate dimension reduced from 2-D to 1-D.
- One unknown curvature coefficient, identifiable after one nonzero forward observation.
- Online 1-D curvature refit from mandatory trial evaluations.
- Hardware-cost-aware failed-trial budget: max(1, floor(measured gradient/forward cost ratio)).
- MVR threshold remains based on measured hardware query cost.
- Safety cap remains 6 accepted moves per gradient for comparability.

## Scientific comparison
The script runs:
1. Momentum
2. Adam
3. AGRLS-v0.7 Adam-anchored 2-D PF-MVR
4. AGRLS-v0.8 Adam-anchored 1-D PF-MVR

All methods use the same initial network state, MNIST subset, seed, and 60 accepted updates.

## Windows CMD
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage1d_adam_1d_pf_mvr_seed0.cmd
```

Or directly:

```cmd
python agrls_stage1_mnist_fullbatch_v08.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --algorithms momentum adam agrls_adam_pf_mvr agrls_adam_1d_pf_mvr --out stage1d_mnist_adam_1d_pf_mvr_seed0
```

## Return to ChatGPT
Zip the entire output folder:
`stage1d_mnist_adam_1d_pf_mvr_seed0`

## Reproducibility note
Seed 0 remains a development trajectory. Do not interpret this run as held-out evidence. Its purpose is mechanistic diagnosis before freezing the neural-network controller.
