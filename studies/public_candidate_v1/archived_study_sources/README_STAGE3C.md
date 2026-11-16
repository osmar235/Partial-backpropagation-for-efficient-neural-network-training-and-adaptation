# AGRLS Stage 3C — frozen v0.11 cap-2 held-out CNN validation

This package validates the frozen CNN controller on untouched Fashion-MNIST seeds 46-54.

## Frozen method
AGRLS-v0.11 keeps the v0.10 Adam-consistent deferred-backward design and changes only the information horizon:

- `max_without_grad = 2`
- one Adam anchor + at most one surrogate continuation per exact-gradient cycle
- no dedicated map probes
- 1-D surrogate
- incremental MVR threshold
- Adam LR = 0.001

## Main run
From Windows CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage3c_v11_cnn_seeds_46_to_54.cmd
```

Equivalent direct command:

```cmd
python agrls_stage3c_v11_cnn_heldout_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --updates 60 --seeds 46 47 48 49 50 51 52 53 54 --device cuda --adam-lr 0.001 --out stage3c_fashionmnist_cnn_v11_cap2_seeds_46_to_54
```

## Preregistered primary criteria
Before inspecting seeds 46-54:

1. median accuracy deficit <= 1.0 percentage point vs Adam;
2. median backward-pass reduction >= 10%;
3. median calibrated pure-NN compute reduction > 0%.

Direct measured F+B time and wall time are secondary engineering outcomes.

## Outputs
The batch runner creates:

- `heldout_raw_summary.csv`
- `heldout_aggregate_summary.csv`
- `heldout_paired_vs_adam.csv`
- `heldout_preregistered_decision.csv`
- held-out accuracy/backward-reduction figure
- calibrated-vs-direct compute figure
- one folder per seed containing the raw traces and summaries.

Do not tune the controller from any individual seed before the full batch is complete.
