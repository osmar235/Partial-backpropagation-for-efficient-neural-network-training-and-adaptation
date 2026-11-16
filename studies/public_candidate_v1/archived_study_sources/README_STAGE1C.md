# AGRLS Stage 1C — Adam-Anchored Probe-Free MVR

**Date:** 2026-10-05  
**Status:** single-seed neural-network diagnostic; not confirmatory evidence.

## Why this test exists

Stage 1B established that eliminating the three dedicated map probes solves the main oracle-cost bottleneck: AGRLS-PF-MVR used only 30 exact-gradient calls and 64 forward-only calls. Its measured oracle time was approximately the same as momentum, but its optimization quality was worse. The remaining problem is therefore not primarily query cost; it is **step quality / conditioning**.

Adam is much stronger on this MNIST MLP. A major reason is coordinate-wise gradient preconditioning. AGRLS-v0.6 used the raw gradient to define its primary subspace direction and therefore did not inherit Adam's coordinate scaling.

Stage 1C tests one focused modification:

> At every exact-gradient refresh, use Adam's preconditioned update as the first local direction and as the first candidate step; then continue with probe-free surrogate moves and the existing marginal-value refresh controller.

No dedicated map probes are reintroduced.

## Algorithms

- momentum SGD
- Adam
- AGRLS-PF-MVR v0.6 (raw-gradient reference)
- AGRLS-Adam-PF-MVR v0.7

## v0.7 refresh cycle

At an exact-gradient query, update Adam moment estimates:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

and define the preconditioned direction

\[
d_t=-\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

The first basis vector is the normalized Adam direction. The second basis vector is the component of raw steepest descent orthogonal to the first direction (with history/random fallback).

The first candidate move after refresh is exactly the Adam-style step

\[
s_t=\eta_{Adam}d_t.
\]

The forward evaluation needed to validate that move is also used as the first curvature observation. Subsequent moves use the probe-free local surrogate. Refreshing remains governed by the measured hardware cost ratio and MVR rule.

## Run

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage1c_adam_pf_mvr_seed0.cmd
```

Direct command:

```cmd
python agrls_stage1_mnist_fullbatch_v07.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --algorithms momentum adam agrls_pf_mvr agrls_adam_pf_mvr --out stage1c_mnist_adam_pf_mvr_seed0
```

## Main questions

1. Does Adam anchoring recover much of Adam's loss decrease while still reducing backpropagation count?
2. How many exact gradients can be skipped without losing late-stage accuracy?
3. Does v0.7 reduce loss faster than raw-gradient v0.6 at equal gradient-query count and measured oracle time?
4. How many trial forwards/rejections are required?
5. Does the remaining Python/vector-management overhead dominate wall time?

## Interpretation rule

Do not move to mini-batch stochastic training unless v0.7 improves the quality/cost tradeoff over v0.6. If the Adam-anchored method remains poor, revisit the local-subspace continuation model rather than adding stochasticity.
