# AGRLS Stage 3D — equal-compute-budget CNN diagnostic

Stage 3C showed a clear split result on held-out Fashion-MNIST CNN seeds 46-54:

- compute/backward savings generalized strongly;
- fixed-60-update accuracy did not.

Because seeds 46-54 have now been inspected, Stage 3D treats them as **development-only seen seeds**.

## Question
At a fixed 60-update count, AGRLS used materially less neural-network compute than Adam. The next question is therefore whether the saved compute can be reinvested into additional AGRLS updates.

For each seed this script compares:

1. `adam`: 60 Adam updates;
2. `agrls_fixed60`: frozen v0.11 cap-2 for 60 accepted updates;
3. `agrls_equal_compute`: the same frozen v0.11 cap-2 controller, but allowed to continue until the calibrated pure forward/backward compute budget of 60 Adam updates is exhausted.

The controller itself is **not retuned**.

## Compute budget
For each seed:

`Adam budget = 60 * (median pure forward cost + median pure backward cost)`

The equal-compute AGRLS run stops before a neural-network operation that would exceed this budget. If the current accepted parameter point has already been obtained and there is not enough remaining budget for a backward plus the next forward, training stops at that accepted point rather than buying a gradient that cannot be used for another update.

## Run
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage3d_equal_compute_seen_seeds_46_to_54.cmd
```

Equivalent direct command:

```cmd
python agrls_stage3d_equal_compute_seen_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --adam-updates 60 --agrls-max-updates 90 --seeds 46 47 48 49 50 51 52 53 54 --device cuda --adam-lr 0.001 --out stage3d_cnn_equal_compute_seen_seeds_46_to_54
```

## Interpretation rule
This is an exploratory development diagnostic, not new held-out evidence. The main question is whether `agrls_equal_compute` closes the accuracy/train-loss gap relative to `agrls_fixed60` while remaining within the Adam calibrated compute budget.

If it does, we will freeze the equal-compute evaluation protocol before using any new CNN seeds.
