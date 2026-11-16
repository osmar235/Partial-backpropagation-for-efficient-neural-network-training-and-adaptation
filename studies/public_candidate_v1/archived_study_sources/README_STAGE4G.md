# Stage 4G — Forward-Gated Gradient Refresh (development)

This stage is a deliberate pivot after the Stage 4F mechanism ablation.

Stage 4F showed that the v0.14 1-D quadratic continuation did **not** provide a clear advantage over simpler same-direction reuse on the already-seen Fashion-MNIST seeds. In addition, the unconditional K=2 baseline is algebraically equivalent to taking one Adam displacement with twice the learning-rate multiplier per fresh gradient, and the monotone K=2 baseline is a binary same-batch step-length controller. Those results weaken the claim that the v0.14 parabolic surrogate itself is the important mechanism.

Stage 4G returns to the project's more distinctive idea: **use forward evidence on the new mini-batch to decide whether the next backward pass is worth paying for.**

## Algorithms

1. `adam_tensorwise`
   - one new mini-batch per update;
   - forward + backward every batch;
   - tensor-wise Adam state, no full-model flattening.

2. `periodic_k2_reuse`
   - one new mini-batch per update;
   - fresh gradient on one batch, then blindly reuse the last Adam displacement on the next batch;
   - the reuse batch still performs one forward so data exposure and NN-query accounting are explicit;
   - maximum one skipped backward in a row.

3. `fggr_monotone`
   - after a fresh gradient, the next new mini-batch is eligible for a forward-first stale-direction trial;
   - compute `f_B(theta)` with its graph retained;
   - evaluate `f_B(theta + d_stale)` by a stateless forward on the **same batch**;
   - if the candidate lowers the same-batch loss, accept it and skip backward;
   - otherwise, backward through the already-paid `f_B(theta)` graph and take a fresh Adam step;
   - after a successful skip, force a fresh gradient on the following batch.

Thus every method processes exactly one new mini-batch and makes exactly one parameter update per cycle. No loss comparison crosses a mini-batch boundary.

## Why this stage is scientifically different from v0.14

The v0.14 mini-batch controller always purchased a fresh gradient at the start of a cycle and then used one extra forward to choose a larger step along that current Adam direction. That behavior is close to stochastic line-search methods such as PAL (NeurIPS 2020), which fit a one-dimensional parabola using the current loss, directional derivative, and one extra loss evaluation.

Stage 4G instead tests whether a **stale exact-gradient-derived direction can be validated forward-only on the next mini-batch before deciding to buy a new backward pass.**

## Status

Seeds 64–72 are already seen. This is a development/mechanism experiment only.

## Run

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4g_forward_gated_seen_seeds_64_to_72.cmd
```

Direct command:

```cmd
python agrls_stage4g_forward_gated_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --updates 400 --seeds 64 65 66 67 68 69 70 71 72 --device cuda --adam-lr 0.001 --out stage4g_forward_gated_seen_seeds_64_to_72
```

Primary development questions:

- Does the same-batch forward gate preserve accuracy better than blind periodic stale-direction reuse?
- What fraction of eligible stale trials are accepted?
- Does the gate still reduce backward passes and calibrated NN compute after paying its extra trial forward?
- Does the tensor-wise implementation reduce controller/wall-time overhead enough to make later scaling tests meaningful?
