# Sampled weight-gradient operation-cost gate — 2026-10-25

The author closed Stage15A and asked to continue. Test a distinct mechanism:
preserve every token, native forward and dense input-gradient propagation;
approximate only MLP linear weight-gradient outer-product sums. All original
model parameters remain trainable. No side network, compressed forward or frozen
prefix. This is an established sampling family, not a new-algorithm claim.

Use artificial full-shape ViTs (six blocks, width 192, 65 tokens, native dropout),
artificial random weights/inputs/labels. Never load real task checkpoints/data.
Native, exact-custom and sampled-quarter arms start from identical tensors.
Exact-custom uses every token; it measures custom-autograd overhead and is a
correctness control. The candidate keeps CLS and 16 of 64 patch positions per
MLP linear layer/update, shared across the images in the batch. Each module has
a separate deterministic private sampling seed. New draws each update; no impact
on dropout RNG. For dW, weight selected patch terms by 4 and CLS terms by 1.
Keep db and dX dense and exact in algebra (floating-point tolerances tested).
Use independent masks for each of the twelve MLP linear modules. Attention,
normalization, embeddings, classifier and nonlinearities retain native gradients.
Inference uses the native linear forward without sampling. First-order only.

Fix two artificial seeds, batches 256 and 200, six cyclic counterbalanced rounds
per condition. Three arms per round: 72 trials. Three discarded warmup updates
per trial, exactly restore all model tensors and fresh AdamW. Measure the first
retained update separately (state allocation), followed by eight steady updates.
Total 216 discarded and 648 retained artificial updates. Match inputs, source
weights, dropout seeds and optimizer hyperparameters. Save raw losses, state
hashes, order, memory, device/telemetry, setup and warmup/restoration time. No sweep.

The synchronized update clock includes seed setup, sampling/index creation,
gathering, complete forward, cross-entropy and scalar readback, all backward
work and AdamW. No hidden gradient precomputation. It excludes input creation,
model setup, warmup, correctness checks, logging, real augmentation, validation
and checkpoints. Report setup/warmup and allocation separately. This is an
operation gate; complete adaptation time and task quality remain unmeasured.

Gate: median of within-round candidate/native steady-update ratios <=0.90 in
each of the four seed/batch conditions. No pass based on exact-custom alone.
Show individual ratios and passing-round counts; timing rounds are not training
replications. Correctness and frozen input/source checks must pass. CPU smoke
is always pipeline validation and cannot pass the scientific cost gate.

A pass permits designing a bounded real gradient-quality/learning screen;
it cannot establish useful adaptation, novelty or AdamW convergence. A failure
parks this fixed implementation for the time objective, without changing the
sampling fraction or thresholds after seeing timings. No automatic next run.

Realization: dW = sum_i g_i x_i^T. CLS is exact; a uniform subset S of k patches
gives sum_CLS + (64/k) sum_S. Its expectation is the full dW, conditional on the
current batch, weights and dropout. Shared masks across images introduce
correlation but not bias. Input VJPs use all tokens, so sampled dW does not alter
upstream error signals within the same backward pass. AdamW's nonlinear update
is not thereby unbiased, and subsequent trajectories can diverge. Variance and
overhead can defeat any benefit. Full-rate gradients and exact enumeration on
a small linear problem must pass before this benchmark is frozen.

Prior art checked: [approximate tensor operations, NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/eaa1da31f7991743d18dadcf5fd1336f-Abstract.html),
[VCAS](https://arxiv.org/abs/2402.17227), and
[TokenTune](https://aclanthology.org/2024.emnlp-main.1202/).
This uniform, weight-only ViT variant is not an exact reproduction of them.
