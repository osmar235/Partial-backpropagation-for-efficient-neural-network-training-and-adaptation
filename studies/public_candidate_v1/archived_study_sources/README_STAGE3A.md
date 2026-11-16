# AGRLS Stage 3A — deterministic CNN compute-scaling diagnostic

Purpose: test frozen AGRLS-v0.10 on a heavier deterministic full-batch neural network before introducing mini-batch stochasticity.

## Model
Fashion-MNIST CNN:
- Conv2d 1->32, 3x3, padding 1; ReLU; MaxPool2d(2)
- Conv2d 32->64, 3x3, padding 1; ReLU; MaxPool2d(2)
- Linear 64*7*7 -> 128; ReLU
- Linear 128 -> 10
- 421,642 trainable parameters
- no dropout or batch normalization

## Development run
- dataset: Fashion-MNIST
- train subset: 10,000
- evaluation subset: 5,000
- updates: 60
- seed: 0 (development only)
- Adam LR: 0.001 for both Adam and AGRLS Adam anchors
- algorithms: Adam and frozen AGRLS-v0.10 Adam-consistent deferred-backward

## Run
From Windows CMD:

    C:\Users\osmar\projects\.venv\Scripts\activate
    cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
    run_stage3a_v10_cnn_seed0.cmd

Direct command:

    python agrls_stage3a_cnn_fullbatch_v10.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --adam-lr 0.001 --out stage3a_fashionmnist_cnn_fullbatch_v10_seed0

Please ZIP the entire output folder and upload it.

## Important
Do not change the model, seed, learning rate, or subset size unless the run fails. If CUDA reports out-of-memory, stop and report the exact error rather than silently changing the subset; the experimental record should be updated before rerunning.
