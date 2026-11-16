# Pet selective development — prospective protocol

Prepared after baseline qualification and before selective development outcomes.
Use six fresh development adaptation seeds 2701–2706, each with full-gradient,
permanently frozen-prefix and periodic-r4 arms. Six seeds permit one occurrence
of each possible three-arm execution order and improve on a three-pair variance
estimate. They remain development seeds, not independent publication evaluation.
Only the existing 2944/736 trainval split is available. No test phase exists here.

All arms use the qualified baseline's pretrained ResNet-18, 224px cached inputs,
15 epochs, batch 64, BF16 autocast, FP32 parameters/moments/loss, fused AdamW,
channels-last, fixed BatchNorm statistics with trainable affine parameters,
deterministic algorithms and the same learning-rate/augmentation recipe.
The new full arm runs alongside both candidates to supply matched timings;
historical baseline timings are not substituted into these comparisons.

The prefix is conv1/bn1/relu/maxpool/layer1/layer2. Suffix is layer3/layer4/
avgpool/flatten/fc. Full uses the original model forward. Inactive-prefix steps
run the prefix with no autograd graph; gradients are None after zero_grad, so
AdamW moments, step age and weight decay pause. Periodic refreshes occur at
zero-based global indices 0,4,8,... (173 refreshes among 690 updates). The frozen
arm never updates prefix weights. Every arm updates the suffix at every step.
BatchNorm buffers never update. No boundary, cadence, loss-gate or LR sweep.

Each paired seed has identical initial backbone/head and epoch sample/crop/flip
streams. All arms have three discarded warmup updates following their own
schedule, then exact model restoration and a new optimizer. The complete clock
starts before cache/model loading and charges initialization, warmup/restoration,
training, epoch common-FP32 validation, final BF16 validation, integrity work
within the trajectory, and checkpoint/logit/log writing. Hash manifests,
source/data preflight, imports and the final independent audit are excluded.
Source pretraining and previous one-time dataset preparation are excluded.
Frozen arms legitimately save optimizer-state initialization/storage costs.
Report training time and complete time separately; one timing per trajectory.

Save final-epoch logits, checkpoints, optimizer state, complete and training
times, streams and step logs. Audit expected optimizer ages, frozen-prefix
identity, fixed BatchNorm buffers, schedule/LR/data streams, initialization,
file hashes, independent NumPy scoring and checkpoint rescoring in both modes.
Completed verified runs skip training; incomplete attempts are preserved. More
than one launch or attempt marks timing fragmented and requires review. An
integrity failure stops the run; an unfavorable accuracy result is retained
and does not select a different schedule. No quality-based early stopping.

After this development run, estimate periodic-minus-full paired accuracy SD and
timing feasibility. Before official-test access, specify the exact variance
uncertainty adjustment, t-based seed calculation for a desired 0.20 pp half-width
(cap 24), independent evaluation seeds, multiplicity and final analysis. The
planned primary margin remains 0.30 pp. Do not pool development with evaluation
or choose evaluation based on a favorable candidate mean. This command stops
for review and cannot launch official-test evaluation.

Synthetic tiny and native-ResNet smoke runs validate execution without examining
real selective outcomes. Source hashes, successful preparation checks and the
qualified baseline evidence are frozen before the real command can run.
