# AGRLS Stage 1B — Probe-Free MVR diagnostic

This package performs a controlled repeat of the first MNIST full-batch neural-network gate.

## What changes from Stage 1A

Legacy AGRLS-MVR reconstructs a 2-D map with three dedicated forward probes after each gradient refresh.

AGRLS-v0.6 **PF-MVR** removes all dedicated map probes. It:

1. obtains the exact local slope from backpropagation (`b = V^T g`),
2. transfers curvature information from the previous subspace,
3. uses a Barzilai-Borwein-like scalar curvature prior for unrepresented directions,
4. updates the surrogate from the forward evaluations already needed to test candidate moves.

The experiment compares:

- SGD
- momentum SGD
- Adam
- legacy AGRLS-MVR
- AGRLS PF-MVR

## Environment

Expected virtual environment:

`C:\Users\osmar\projects\.venv`

Project folder:

`C:\Users\osmar\projects\Projeto_AI_Fronteira`

## Run

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage1b_pf_mvr_seed0.cmd
```

Equivalent direct command:

```cmd
python agrls_stage1_mnist_fullbatch_v06.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --algorithms sgd momentum adam agrls_mvr agrls_pf_mvr --out stage1b_mnist_fullbatch_pf_mvr_seed0
```

## What to upload

Zip the entire output folder:

`stage1b_mnist_fullbatch_pf_mvr_seed0`

Do not run multiple seeds yet. This is still a controlled algorithm-development ablation.
