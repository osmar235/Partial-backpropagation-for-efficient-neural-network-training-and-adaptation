# AGRLS Stage 5B — projected-gradient Adam state

## Why this branch exists

Stages 5A7 and 5A10 reduced measured neural-network time by 18.05% and 18.86%,
respectively, but lost 6.85 and 8.05 evaluation-accuracy points. The skipped
updates moved parameters without advancing Adam's first moment, second moment,
or bias-correction clock. Stage 5B tests whether a gradient reconstructed from
current-batch forward probes can keep those state transitions synchronized.

This document is a development protocol. It does not assert that the method is
faster or more accurate than Adam.

## Mechanism

Let the columns of `D` be the four most recent exact-Adam displacements and let
`G = D^T D`. One-sided current-batch probes estimate

```text
s_i = [L(theta + r d_i) - L(theta)] / r ~= d_i^T g.
```

The ridge-regularized minimum-norm gradient in this subspace is

```text
g_projected = D (G + lambda I)^-1 s,
lambda = ridge * trace(G) / 4.
```

The proposed skipped-backward transition previews the ordinary Adam equations:

```text
m_next = beta1*m + (1-beta1)*g_projected
v_next = beta2*v + (1-beta2)*g_projected^2
d_next = -lr * bias_correct(m_next) / (sqrt(bias_correct(v_next)) + eps).
```

A disjoint probe accepts or rejects `d_next`. The trial displacement is capped
at the median norm of the four historical Adam steps. In a future live
controller, acceptance would commit `m_next`, `v_next`, and the Adam clock;
rejection would discard the preview and use the exact gradient instead.

Stage 5B1 keeps the live trajectory on ordinary Adam. Its preview is read-only,
and the exact gradient supplies diagnostic truth. This makes Stage 5B1 a
falsification screen, not evidence of final optimizer performance.

## Frozen Stage 5B1 screen

- CIFAR-10 development seed 0 only
- ResNet-18 with GroupNorm, batch 256, 500 updates
- four exact-Adam history directions
- 32-example fit probe and disjoint 32-example confirmation probe
- one-sided radius 0.125, ridge `1e-4`
- seven probe forwards per eligible attempt
- trust-radius ratio 1.0

Proceed to one on-policy controller test only if all four conditions hold:

1. full-batch true descent is at least 75%;
2. confirmation precision is at least 90%;
3. acceptance exceeds the calibrated neural-network break-even rate;
4. the median capped-step/exact-Adam-step norm ratio lies in `[0.5, 1.5]`.

The later controller must still finish within 1 percentage point of Adam,
reduce backward passes by at least 10%, and show positive calibrated
neural-network savings before any untouched-seed run.

## Cost boundary

Seven 32-example forwards are the neural-network query cost. Reconstructing a
full gradient and previewing Adam add `O(4P)` tensor reads plus full-size moment
arithmetic for `P` parameters. The Python prototype measures that overhead but
does not optimize it. A wall-clock claim requires flat buffers or fused kernels
and direct end-to-end timing; calibrated query savings alone are insufficient.

## Prior-art boundary

Using estimated gradients to update Adam-like moments is established in
zeroth-order adaptive optimization. Randomized and history-informed subspace
gradient estimation is also an active line of work. The research question here
is narrower: whether intermittent exact-backprop anchors, a historical Adam-step
subspace, a disjoint acceptance gate, and a synchronized adaptive-state
transition can save end-to-end training time without losing learning progress.

## Run

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage5b1_cifar10_projected_adam_diagnostic_seed0.cmd
```

## Stage 5B1 result

The frozen 500-update CIFAR-10 seed-0 diagnostic completed on the RTX 5080.
It failed every quality criterion and also failed the calibrated query-cost
criterion.

| metric | result | required |
|---|---:|---:|
| full-batch true descent | 37.50% | >=75% |
| confirmation precision | 70.83% | >=90% |
| gate acceptance | 38.71% | >41.76% break-even |
| median capped/exact step norm | 1.004 | 0.5–1.5 |

The norm criterion passed only because the trust radius clipped 96.77% of raw
previews. Before clipping, the median projected-Adam step was 2.256 times the
exact Adam step. The median reconstructed/exact gradient cosine was 0.0794 and
the median relative error was 0.9986, so the four-direction history did not
recover enough of the current gradient to support an Adam state transition.

The failure persisted rather than disappearing after warm-up:

| updates | true descent | gate acceptance | precision | median gradient cosine |
|---:|---:|---:|---:|---:|
| 5–100 | 53% | 53% | 76% | 0.05 |
| 101–200 | 43% | 38% | 82% | 0.09 |
| 201–300 | 32% | 38% | 61% | 0.10 |
| 301–400 | 34% | 31% | 74% | 0.08 |
| 401–500 | 26% | 34% | 59% | 0.09 |

The modeled neural-network saving was already negative at -3.05% before
counting reconstruction overhead. The Python reconstruction and Adam preview
cost 40.74 ms per attempt on average, which is nearly the calibrated cost of a
full forward plus backward (44.56 ms). Therefore this exact mechanism cannot
beat Adam on either quality or wall time, and no live Stage 5B controller should
be run.

## Decision

Close the full-parameter projected-gradient branch. Do not tune its ridge,
probe radius, trust radius, or history length on seed 0. The next defensible
mechanism test should obtain real current gradients for a selected parameter
block, because the historical four-direction span is too small to reconstruct
the full gradient. A blockwise test must first measure actual backward time for
candidate blocks; FLOP intuition alone is not enough on this GPU.
