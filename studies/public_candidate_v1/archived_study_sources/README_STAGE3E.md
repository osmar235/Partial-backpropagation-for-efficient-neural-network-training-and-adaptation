# AGRLS Stage 3E — held-out equal-compute CNN validation

Purpose: validate the **frozen AGRLS-v0.11 cap-2 controller** under the equal-calibrated-compute evaluation protocol developed in Stage 3D.

## Scientific status
- Seeds 46-54 are development/seen and are **not** used here.
- Seeds **55-63** are reserved held-out seeds.
- The AGRLS controller is unchanged from v0.11.
- Only the evaluation/stopping rule differs from fixed-update experiments: AGRLS may take additional accepted updates until the next required neural-network operation would exceed Adam's calibrated pure-NN compute budget.

## Preregistered primary criteria
Across seeds 55-63:
1. median accuracy deficit vs Adam <= 1.0 percentage point;
2. median backward-pass reduction >= 10%;
3. median calibrated budget use <= 100% of Adam.

## Windows CMD
From the project directory:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage3e_equal_compute_heldout_seeds_55_to_63.cmd
```

Equivalent direct command:

```cmd
python agrls_stage3e_equal_compute_heldout_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --adam-updates 60 --agrls-max-updates 90 --seeds 55 56 57 58 59 60 61 62 63 --device cuda --adam-lr 0.001 --out stage3e_cnn_equal_compute_heldout_seeds_55_to_63
```

When complete, ZIP the entire output folder `stage3e_cnn_equal_compute_heldout_seeds_55_to_63` and upload it.
