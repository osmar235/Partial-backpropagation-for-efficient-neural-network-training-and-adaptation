# AGRLS Stage 4A — v0.12 batch-locked mini-batch bridge

This is a **development diagnostic** for stochastic mini-batch training. It does not change the frozen deterministic evidence and does not use a new held-out seed.

## Scientific purpose
The deterministic AGRLS surrogate relies on comparing loss changes along a local parameter-space direction. With stochastic mini-batches, losses from different batches are not directly comparable. Stage 4A therefore locks every local AGRLS cycle to one mini-batch: the exact gradient, Adam anchor evaluation, and at most one surrogate continuation all use the same batch. Surrogate state is discarded at the batch boundary.

This asks a clean first stochastic question: **can one mini-batch backward support two useful parameter updates without mixing losses from different batches?**

## Default experiment
- Fashion-MNIST CNN (`SmallFashionCNN_32_64_128`, 421,642 parameters)
- train subset 10,000; eval subset 5,000
- batch size 256, drop incomplete tail
- Adam LR 0.001
- seed 0 (development only)
- Adam: 400 updates
- AGRLS fixed-update: 400 accepted updates
- AGRLS equal-compute: same calibrated pure-NN budget as Adam, max 700 accepted updates

## Run
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4a_v012_minibatch_locked_seed0.cmd
```

Direct command:
```cmd
python agrls_stage4a_minibatch_locked_v012.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --adam-updates 400 --agrls-max-updates 700 --seed 0 --device cuda --adam-lr 0.001 --out stage4a_fashionmnist_cnn_minibatch_locked_seed0
```

## Expected outputs
- `summary.csv`
- `paired_vs_adam.csv`
- `trace_adam.csv`
- `trace_agrls_batchlocked_fixed.csv`
- `trace_agrls_batchlocked_equal_compute.csv`
- `final_accuracy_vs_backward_passes.png`
- `final_accuracy_vs_calibrated_compute.png`
- `REPORT.txt`

## Important interpretation rule
The mini-batch loss in one cycle is only compared with other losses from the **same mini-batch**. Do not interpret batch-local losses as a smooth global training-loss curve across cycles. Also report how many mini-batch cycles were consumed: AGRLS may take two parameter updates on one batch, so equal compute does not imply equal numbers of distinct batch presentations.
