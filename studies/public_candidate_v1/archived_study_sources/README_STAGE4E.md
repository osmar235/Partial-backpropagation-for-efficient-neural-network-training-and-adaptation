# AGRLS Stage 4E — frozen v0.14 stochastic held-out validation

This package validates the frozen AGRLS-v0.14 controller on untouched Fashion-MNIST stochastic seeds **73–81**.

## Frozen algorithm
- SmallFashionCNN_32_64_128 (421,642 parameters)
- train subset 10,000; eval subset 5,000
- mini-batch size 256
- Adam LR 0.001
- batch-locked cycles
- fresh exact gradient at each new batch cycle
- Adam-consistent anchor
- 1-D information-matched surrogate
- cap = 2 accepted updates per batch cycle
- no terminal continuation-verification forward
- no cross-batch surrogate state
- continuation requires monotone anchor + finite positive predicted gain
- post-anchor MVR threshold = **0.0** (zero marginal NN-query cost)

## Held-out seeds
73 74 75 76 77 78 79 80 81

Do not tune the controller using these seeds.

## Run on Windows CMD
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4e_v014_minibatch_seeds_73_to_81.cmd
```

Equivalent direct command:
```cmd
python agrls_stage4e_v014_minibatch_heldout_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --adam-updates 400 --agrls-max-updates 700 --seeds 73 74 75 76 77 78 79 80 81 --device cuda --adam-lr 0.001 --out stage4e_fashionmnist_cnn_minibatch_v014_seeds_73_to_81
```

The script writes per-seed traces, paired summaries, aggregate summaries, a preregistered-decision CSV, a report, and diagnostic plots.

A tiny fake-data CPU smoke test was used only to check code paths and aggregation. No Fashion-MNIST result for seeds 73–81 was generated during package preparation.
