# AGRLS Stage 5A2 — CIFAR-10 FGGR development diagnostics

## Why Stage 5A2 was run

Stage 5A1 was technically stable and preserved most of Adam's accuracy, but it
did not produce an economical controller on CIFAR-10. At 1,500 updates, the
probe-128 gate accepted only 69 of 1,430 stale trials (4.83%), versus a measured
break-even acceptance rate of 28.68%. Calibrated neural-network compute was
22.74% higher than Adam, although evaluation accuracy was only 0.47 percentage
points lower.

The Stage 5A1 trace also showed that acceptance was concentrated at the start:
34% during updates 1–100, 10% during 101–200, and approximately 1–5% in most
later 100-update windows.

Stage 5A2 therefore used the already-seen development seed 0. It is diagnostic
development evidence, not held-out confirmation.

## Stage 5A2A: stale-displacement scale screen

The model, data, batch size, probe size, Adam learning rate, and gate were kept
fixed. Only the fraction of the stale Adam displacement was changed. Each run
used 500 accepted updates and probe size 128.

| stale scale | eval accuracy | delta vs Adam | backward reduction | calibrated compute reduction | acceptance | break-even |
|---:|---:|---:|---:|---:|---:|---:|
| Adam | 60.79% | — | — | — | — | — |
| 0.125 | 56.96% | -3.83 pp | 19.60% | -3.41% | 24.38% | 28.62% |
| 0.25 | 57.63% | -3.16 pp | 14.40% | -10.04% | 16.86% | 28.62% |
| 0.5 | 60.64% | -0.15 pp | 11.00% | -14.42% | 12.39% | 28.62% |
| 1.0 | 56.68% | -4.11 pp | 8.20% | -18.02% | 8.95% | 28.62% |

Smaller stale steps increased acceptance, but no scale crossed the hardware
break-even rate. Scale 0.5 was the only tested setting that closely preserved
Adam accuracy, and it still used 14.42% more calibrated neural-network compute.

## Stage 5A2B: probe-size check at scale 0.5

Because scale 0.5 preserved accuracy best, it was held fixed while probe size
was reduced. Each run again used 500 updates on seed 0. The Adam accuracy
reference is the Stage 5A2A run above.

| probe size | eval accuracy | delta vs Adam | backward reduction | calibrated compute reduction | acceptance | break-even |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 55.44% | -5.35 pp | 14.60% | -5.26% | 17.10% | 23.31% |
| 32 | 57.08% | -3.71 pp | 17.40% | +5.33% | 21.10% | 14.65% |
| 16 | 53.58% | -7.21 pp | 21.40% | +10.55% | 27.30% | 13.84% |

Smaller probes can make the gate economical, but their decisions degrade the
optimization result substantially. No tested point simultaneously preserved
accuracy and reduced calibrated compute.

## Decision after Stage 5A2

Do not move this cross-batch stale-displacement controller to untouched CIFAR-10
seeds. Do not spend a full 1,500-update run on any Stage 5A2 candidate. The
current evidence shows two distinct failure modes:

1. A sufficiently reliable probe is too expensive relative to the low rate at
   which the previous mini-batch's Adam displacement remains useful.
2. A small probe lowers the break-even threshold but introduces enough gate
   error to cause a large accuracy loss.

The next useful experiment is an instrumentation-only viability diagnostic
during ordinary Adam training. It should measure consecutive-mini-batch gradient
alignment, finite-step full-batch descent, and probe-gate precision/recall over
training. That diagnostic can distinguish true stale-direction failure from
probe misclassification before another controller is designed. Untouched seeds
should remain untouched until a development mechanism has both an accuracy and
compute case.

## Reproduce Stage 5A2A

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage5a2a_cifar10_fggr_scale_screen_seed0.cmd
```

The implementation records current and trial probe losses in every FGGR trace:
`agrls_stage5a2_cifar10_fggr_scale_screen.py`.

Follow-on viability, direction-bank, and live-controller results are recorded in
`README_STAGE5A3_TO_STAGE5A5.md`.
