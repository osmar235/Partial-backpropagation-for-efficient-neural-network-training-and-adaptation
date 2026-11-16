# AGRLS Stage 4D — v0.14 zero-marginal-NN-cost continuation

This is a **development experiment on seen Fashion-MNIST seeds 64–72**. It is not confirmatory evidence.

## Why this stage exists
Stage 4C showed that frozen v0.13 passed the equal-compute held-out criteria, but the fixed-400-update condition missed the calibrated-compute criterion by a small margin. v0.13 still uses the conservative MVR threshold inherited from the earlier verified-continuation design.

After the same-batch Adam-anchor forward has already been paid, v0.13's one allowed continuation is committed without another forward or backward. Its marginal NN-query cost is therefore zero. v0.14 makes one change only:

- v0.13: `mvr_threshold = 1 / ((F+B)/F)`
- v0.14: `mvr_threshold = 0.0`

A continuation is still allowed only after a monotone Adam anchor and only when the 1-D model predicts a finite positive improvement. The surrogate is discarded immediately after the continuation; a new mini-batch always starts with a fresh exact gradient.

## Run on Windows CMD
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4d_v014_zero_cost_seen_seeds_64_to_72.cmd
```

Equivalent direct command:
```cmd
python agrls_stage4d_v014_seen_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --adam-updates 400 --agrls-max-updates 700 --seeds 64 65 66 67 68 69 70 71 72 --device cuda --adam-lr 0.001 --out stage4d_v014_zero_cost_seen_seeds_64_to_72
```

The batch compares Adam, v0.13 fixed-400, v0.14 fixed-400, and v0.14 equal-compute with rotated execution order.

No scientific Fashion-MNIST Stage 4D result was generated in the preparation environment. Only syntax checks and tiny fake-data CPU smoke tests were run.
