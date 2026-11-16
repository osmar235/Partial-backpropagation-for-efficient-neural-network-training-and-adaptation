# AGRLS Stage 5A0 — ResNet-18-GN microprobe economics calibration

## Why this comes before CIFAR-10 training
Stage 4H showed that a 32-example forward gate was **informative** but not **economical** on the small Fashion-MNIST CNN: it preserved near-Adam accuracy and improved accuracy relative to blind stale-gradient reuse, but the observed stale-reuse acceptance rate (~0.33 median) was below the measured hardware break-even acceptance rate (~0.55 median).

The likely reason is systems-level: on the small CNN, a 32-example GPU forward cost almost as much as a 256-example forward because fixed GPU/kernel overhead dominates.

Before spending time training the 11.17M-parameter CIFAR model, Stage 5A0 asks one narrow question only:

> On the much larger CIFAR-adapted ResNet-18-GN, does a small probe forward become cheap enough relative to a full 256-example forward+backward that forward-gated gradient refresh can plausibly save compute?

## What this script does
It uses **synthetic 32x32 RGB tensors only**. It does not load CIFAR-10 and does not observe any CIFAR training outcome.

It measures on the local GPU:
- median full-batch forward time at batch 256;
- median full-batch backward time at batch 256;
- median no-grad probe-forward time for probe sizes 8, 16, 32, 64, and 128;
- a systems diagnostic for the stateless trial forward including candidate-parameter construction;
- the Stage-4H-style calibrated break-even acceptance rate:

`p_break_even = 2 * F_probe / (F_full + B_full)`

The timing-only selection rule is frozen before observing CIFAR outcomes:

> choose the **largest** tested probe size whose calibrated break-even acceptance rate is <= 0.30.

The 0.30 threshold is deliberately conservative relative to the Stage 4H seen-data microprobe acceptance range (~0.30–0.36). If no tested size meets it, do not launch the CIFAR training run yet.

## Run
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage5a0_resnet18gn_probe_economics.cmd
```

Direct command:
```cmd
python agrls_stage5a0_resnet18gn_probe_economics.py --device cuda --batch-size 256 --probe-sizes 8 16 32 64 128 --warmup 8 --reps 25 --target-break-even 0.30 --out stage5a0_resnet18gn_probe_economics
```

Send back the resulting `stage5a0_resnet18gn_probe_economics` folder or ZIP. The key files are `REPORT.txt`, `probe_economics.csv`, and `calibration_meta.json`.
