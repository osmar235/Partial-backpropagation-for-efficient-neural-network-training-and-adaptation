# AGRLS Stage 2C — frozen v0.10 held-out Fashion-MNIST validation

## Purpose
Validate the Adam-consistent anchor change on completely untouched Fashion-MNIST seeds 37–45.

## Frozen algorithm
AGRLS-v0.10 Adam-consistent deferred-backward controller. No tuning on seeds 37–45.

## Fixed experimental design
- Dataset: Fashion-MNIST
- Model: MLP 784 -> 128 -> 64 -> 10
- Train subset: 20,000
- Eval subset: 5,000
- Accepted updates: 60
- Adam LR: 0.01
- Baseline: Adam
- Seeds: 37–45
- Execution order alternates by seed
- Pure forward/backward calibration is used for the common compute estimate

## Preregistered primary criteria
Supportive if the paired medians satisfy all three:
1. evaluation accuracy deficit <= 1.0 percentage point vs Adam;
2. backward-pass reduction >= 10%;
3. calibrated pure neural-network compute reduction > 0%.

Direct instrumented NN time and total wall time remain secondary engineering outcomes.

## Run
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage2c_v10_fashion_seeds_37_to_45.cmd
```

Or:
```cmd
python agrls_stage2c_v10_fashion_heldout_batch.py --dataset fashionmnist --train-subset 20000 --eval-subset 5000 --updates 60 --seeds 37 38 39 40 41 42 43 44 45 --device cuda --out stage2c_fashionmnist_fullbatch_v10_seeds_37_to_45
```

Zip the complete output folder and upload it for analysis.
