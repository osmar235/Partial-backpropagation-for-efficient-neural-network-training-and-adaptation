# AGRLS Stage 4C — frozen v0.13 stochastic mini-batch held-out validation

This package freezes the Stage 4B AGRLS-v0.13 controller and evaluates it on untouched Fashion-MNIST CNN seeds **64-72**.

Frozen method:
- batch size 256;
- Adam LR 0.001;
- batch-locked local cycles;
- one exact mini-batch gradient per cycle;
- at most one surrogate continuation after the Adam anchor;
- no terminal continuation-verification forward;
- no surrogate state crosses a mini-batch boundary.

Two operating points are evaluated against Adam (400 updates):
1. `agrls_fixed`: 400 accepted updates;
2. `agrls_equal_compute`: continue up to 700 updates but stop before exceeding Adam's calibrated pure mini-batch NN-compute budget.

Preregistered fixed-update criteria:
- median accuracy delta >= -1.0 pp;
- median backward reduction >= 10%;
- median calibrated NN-compute reduction > 0%.

Preregistered equal-compute criteria:
- median accuracy delta >= -1.0 pp;
- median backward reduction >= 10%;
- median calibrated budget use <= 100%.

Run on Windows CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4c_v013_minibatch_seeds_64_to_72.cmd
```

Direct command:

```cmd
python agrls_stage4c_v013_minibatch_heldout_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --adam-updates 400 --agrls-max-updates 700 --seeds 64 65 66 67 68 69 70 71 72 --device cuda --adam-lr 0.001 --out stage4c_fashionmnist_cnn_minibatch_v013_seeds_64_to_72
```

Do not tune v0.13 using seeds 64-72 after inspecting these results. If the held-out criteria fail, record the failure and use a new development seed block for any redesign.
