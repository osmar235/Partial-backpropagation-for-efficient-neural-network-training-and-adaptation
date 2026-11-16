# AGRLS Stage 5A1 — CIFAR-10 / ResNet-18-GN Forward-Gated Gradient Refresh

## Purpose
This is a **development/transfer** experiment, not held-out evidence. It tests whether the Forward-Gated Gradient Refresh mechanism becomes economically useful when moved from the small Fashion-MNIST CNN to an 11.17M-parameter CIFAR-10 residual network.

## Frozen hardware-only choice from Stage 5A0
Stage 5A0 used synthetic RGB tensors only. The predeclared selection rule was: choose the **largest tested probe size** whose calibrated break-even acceptance rate is <= 0.30. On the RTX 5080, all tested probe sizes passed and the rule selected **probe size 128**. No CIFAR-10 training outcomes were used to choose this value.

## Algorithms
1. `adam_tensorwise`: standard Adam, one full-batch forward+backward per accepted update.
2. `fggr_microprobe128`: after a fresh Adam update, the next mini-batch is eligible for a 128-example same-batch current/trial forward gate. If the stale displacement improves probe loss, accept it and skip full F+B. If not, restore parameters and pay full F+B. Maximum one consecutive skipped backward.
3. `periodic_k2_noquery`: strong blind reuse baseline; alternates fresh F+B and blind reuse of the last Adam displacement with no NN query on reuse steps.

No loss comparison crosses mini-batch boundaries. GroupNorm is used so extra probe forwards cannot mutate normalization running statistics.

## Protocol
- CIFAR-10 full train/test: 50,000 / 10,000.
- No augmentation in this first transfer test.
- ResNet-18 CIFAR stem + GroupNorm; ~11.17M parameters.
- Batch 256; probe 128.
- Adam LR 0.001.
- 1,500 accepted updates.
- Development seed 0.

## Main questions
- Does observed FGGR probe acceptance exceed the measured break-even rate?
- Does FGGR have positive calibrated/direct NN compute savings vs Adam?
- Does forward gating preserve more accuracy than blind periodic reuse?
- Does larger-model compute make controller overhead a smaller fraction of wall time?

Do not tune probe size or controller thresholds from this run before recording the result.
