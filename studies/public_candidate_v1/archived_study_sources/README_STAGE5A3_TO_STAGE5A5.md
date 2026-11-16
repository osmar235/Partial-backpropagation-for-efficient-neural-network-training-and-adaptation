# AGRLS Stages 5A3–5A5 — stale-direction viability and direction-bank decision

## Scope

These stages used the already-seen CIFAR-10 development seed 0. They are
mechanism and feasibility evidence, not held-out confirmation. No untouched
CIFAR-10 seed was used.

## Stage 5A3: is the latest stale Adam displacement viable?

Stage 5A3 followed an ordinary 500-update Adam trajectory. Extra no-gradient
forwards measured the previous Adam displacement on the next mini-batch without
changing the training trajectory.

Main result:

- the stale displacement was an infinitesimal descent direction on only 18.64%
  of new mini-batches;
- median descent cosine was -0.030657, so the typical stale direction was
  slightly uphill on the new mini-batch;
- true full-batch finite-step descent rates were 15.83%, 13.63%, 10.82%, and
  6.01% at scales 0.125, 0.25, 0.5, and 1.0 respectively.

At scale 0.125:

| probe | gate acceptance | precision | recall | break-even | modeled net saving |
|---:|---:|---:|---:|---:|---:|
| 16 | 28.86% | 38.19% | 69.62% | 11.71% | +17.15% |
| 32 | 25.05% | 41.60% | 65.82% | 12.37% | +12.68% |
| 64 | 20.44% | 56.86% | 73.42% | 22.29% | -1.85% |
| 128 | 17.64% | 72.73% | 81.01% | 27.44% | -9.81% |

Small probes look economical only because most accepted choices are false
positives. Reliable probes cost more than the backward computation they can
avoid at the observed true-descent rate.

## Stage 5A4: does a tiny history bank contain a useful direction?

Stage 5A4 retained the last one, two, or four exact-gradient-derived Adam
displacements while keeping the live trajectory as ordinary Adam. The fixed
trial scale was 0.125.

| history | probe | full-batch oracle rate | gate acceptance | selection precision | break-even | modeled net saving |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16.63% | 27.45% | 35.77% | 13.81% | +13.65% |
| 1 | 32 | 16.63% | 23.45% | 42.74% | 14.38% | +9.07% |
| 2 | 16 | 19.68% | 38.76% | 26.94% | 20.71% | +18.04% |
| 2 | 32 | 19.68% | 32.53% | 37.65% | 21.57% | +10.96% |
| 4 | 16 | 61.09% | 76.01% | 48.01% | 34.52% | +41.49% |
| 4 | 32 | 61.09% | 69.76% | 58.96% | 35.95% | +33.81% |

The four-direction bank established a real mechanistic fact: older directions,
especially the third and fourth most recent, often contained a full-batch
descent direction when the newest direction did not. Tensor auditing confirmed
that every stored displacement was independently cloned; the result was not a
state-aliasing artifact.

However, the modeled savings assumed that false-positive selections were
harmless. Stage 5A5 tested that assumption on a live trajectory.

## Stage 5A5: live direction-bank controller

The live controller kept four Adam displacements, used a fixed scale of 0.125,
selected the lowest-loss candidate on the probe, and forced a fresh gradient
after every accepted reuse.

### Probe 32

| metric | Adam | direction bank |
|---|---:|---:|
| evaluation accuracy | 60.79% | 48.92% |
| backward passes | 500 | 299 |
| calibrated NN compute | 21.434 s | 17.382 s |
| reuse acceptance | — | 68.1% |

Paired effects:

- accuracy delta: **-11.87 percentage points**;
- backward reduction: **40.20%**;
- calibrated compute reduction: **18.91%**;
- direct measured NN-time reduction: **20.72%**;
- wall-time reduction: **19.16%**.

### Conservative stop test: probe 64

Only the probe size was increased. The stop criterion required accuracy within
1 percentage point of Adam together with positive calibrated savings.

| metric | Adam | direction bank |
|---|---:|---:|
| evaluation accuracy | 60.79% | 50.16% |
| backward passes | 500 | 305 |
| calibrated NN compute | 21.303 s | 20.842 s |
| reuse acceptance | — | 64.8% |

Paired effects:

- accuracy delta: **-10.63 percentage points**;
- backward reduction: **39.00%**;
- calibrated compute reduction: **2.16%**;
- direct measured NN-time reduction: **6.95%**;
- wall-time reduction: **4.16%**.

The stop test failed decisively on accuracy.

## Decision

Pause the cross-mini-batch Adam-displacement reuse branch. Do not:

- run this controller for 1,500 updates;
- move it to untouched CIFAR-10 seeds;
- tune additional probe sizes, scales, or bank lengths on seed 0;
- interpret the Stage 5A4 oracle result as optimizer performance.

The measurement-first sequence was useful because it separated three facts:

1. The newest stale direction is usually not a descent direction on the next
   stochastic mini-batch.
2. A four-direction bank often contains a descent direction.
3. Cheap forward probes cannot select those directions precisely enough to
   preserve the optimization trajectory; increasing probe reliability consumes
   nearly all of the compute saving and still leaves a large accuracy deficit.

The next research phase should change how a current-batch-informed direction is
obtained, rather than continue tuning a selector over old Adam displacements.
Promising categories to specify and compare with recent prior art are partial or
blockwise backward refresh, forward-mode directional derivatives in a tiny
subspace, and a two-stage cheap-screen/strong-confirmation query. Any new branch
should first receive a cost model and a development-only falsification test like
Stages 5A3–5A5 before using untouched seeds.

The current-batch subspace-combination follow-up is recorded in
`README_STAGE5A6_TO_STAGE5A10.md`.

## Reproduce

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage5a3_cifar10_stale_viability_seed0.cmd
run_stage5a4_cifar10_direction_bank_seed0.cmd
run_stage5a5_cifar10_direction_bank_controller_seed0.cmd
```

The probe-64 stop test used the same Stage 5A5 script with
`--probe-size 64 --out stage5a5b_cifar10_direction_bank_probe64_seed0`.
