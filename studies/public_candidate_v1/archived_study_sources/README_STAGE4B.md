# AGRLS Stage 4B — v0.13 terminal continuation-verification diagnostic

This is a **development-only** Fashion-MNIST mini-batch experiment on seed 0.

Stage 4A showed that AGRLS-v0.12 could reuse one mini-batch gradient safely enough to reduce backward passes, but it paid a terminal same-batch forward to verify every proposed continuation. On seed 0 all 106 continuation probes were accepted, so those forwards did not change the parameter trajectory.

AGRLS-v0.13 changes only one rule: after the post-anchor same-batch forward fits the 1-D local model and the frozen MVR rule approves a continuation, the continuation is committed without a second terminal verification forward. The surrogate is immediately discarded and the next mini-batch begins with a fresh exact gradient. Nothing crosses a mini-batch boundary.

The script compares Adam, the v0.12 verified fixed-update control, v0.13 fixed-update, and v0.13 equal-compute.

## Run

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4b_v013_minibatch_terminal_probe_seed0.cmd
```

Equivalent direct command:

```cmd
python agrls_stage4b_minibatch_terminal_probe_v013.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --adam-updates 400 --agrls-max-updates 700 --seed 0 --device cuda --adam-lr 0.001 --out stage4b_fashionmnist_cnn_minibatch_terminal_probe_seed0
```

Do not use this development seed for later held-out claims.
