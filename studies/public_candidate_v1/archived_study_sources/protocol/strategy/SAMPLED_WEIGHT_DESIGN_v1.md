# Next direction: exact error propagation, sampled weight gradients — 2026-10-25

The author explicitly agreed to close Stage15A and continue. Keep that fixed
side-network recipe closed. Its measured savings and failed quality rule remain
unchanged in the Stage15A review and paper memory. No additional side widths,
taps, learning rates, epochs or seeds are selected.

## Why this distinct mechanism

Our full and head-then-full controls improved substantially; the tested frozen
backbone readouts did not. Token compression also lost accuracy. These outcomes
motivate preserving the original forward and allowing all backbone weights to
learn. They do not prove that any single architectural constraint caused failure.

The next selected candidate approximates one expensive part of backpropagation:
the weight-gradient calculation inside the existing ViT's MLP linear layers.
Every token still participates in the full forward and the dense error signal
passed back to preceding layers. Attention, normalization, embeddings, classifier,
bias gradients, dropout and the architecture stay native. No side parameters or
new inference path. This is different from earlier prefix freezing, stale gradients,
local losses, token merging and side adaptation.

| Alternative | Decision for this step |
|---|---|
| Continue the side recipe | Closed by the author; both seeds failed quality |
| More merging/reset tuning | Parked; would revisit already explored parameters |
| Sample both input and weight gradients | Reserve; first isolate weight sampling while preserving dense credit propagation |
| Weight-gradient sampling with full forward/input derivatives | Selected for a bounded artificial feasibility check |
| Broader real-data sweeps or another architecture immediately | Defer until a mechanism survives correctness and cost checks |

## Mathematical intervention

For a linear layer with Y = X W^T + b and output derivative G:

- Forward: compute the ordinary full Y.
- Input derivative: dX = G W, using all token rows.
- Bias derivative: db = sum G, using all rows.
- Weight derivative: approximate dW = G^T X by a weighted subset of row outer products.

Always retain CLS. Uniformly choose 16 of 64 patch positions per MLP linear
layer per update, shared across the images in the batch. Multiply sampled patch
terms by 64/16 = 4; CLS terms have weight 1. All twelve MLP linear layers have
their own deterministic sampling seed; the draws change each update and consume
no global dropout RNG. The shared spatial mask introduces correlation across
images but leaves the estimator's expectation unchanged.

Conditional on the same current weights, batch and dropout, the expected dW is
the full dW. This follows by linearity and each patch's inclusion probability
16/64. Exact dX means the sampled dW does not corrupt other layers' input error
signals during that same backward pass. Floating-point reductions still require
tolerance checks. AdamW is nonlinear: unbiased dW does not imply an unbiased
AdamW update or a matching training trajectory. Gradient variance may hurt quality.

Only one quarter-patch candidate is selected before full-shape timing. The 16
patch terms plus CLS retain 17/65 token rows for weight outer products. This is
not a promise of 74% less training compute: all forward, all dX and the remaining
model work still execute. Gathering/index generation and Python/custom-autograd
overhead may outweigh arithmetic savings, especially for this small ViT.

## Prior art checked

[Adelman et al., NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/eaa1da31f7991743d18dadcf5fd1336f-Abstract.html)
study sampling-based approximations of tensor operations for faster training.
Their [official code](https://github.com/adelmanm/approx) includes backward-only
sampling options. This establishes direct prior art for the broad construction.

[VCAS](https://arxiv.org/abs/2402.17227) samples activation and weight-gradient
calculations with variance control. Our fixed uniform, weight-only rule does
not implement its adaptive sampling or establish comparable performance.

[TokenTune, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1202/) restricts
backpropagation to selected tokens to reduce activation-memory requirements.
Our prototype keeps dense input-gradient propagation and samples only MLP
weight-gradient sums. That distinction is a controlled experimental choice,
not evidence of originality. No exact reproduction or new-algorithm claim.

Sources were checked on 2026-10-25. No exhaustive novelty search was performed.

## Completed correctness checks

Six tests passed on CPU and CUDA, using only artificial inputs/weights:

1. Exhaustive enumeration of every 2-of-4 patch subset on a small linear layer
   recovers its full weight gradient in expectation; input/bias derivatives agree.
2. The all-token custom derivative passes numerical gradcheck.
3. Full-shape train/eval forwards and global CPU/CUDA RNG states match native.
4. Sampled MLP weights change while input and unsampled parameter gradients match.
5. All-token network gradients match native within the fixed numerical tolerances.
6. Sampling seeds replay exactly, changing seeds changes the sampled gradients,
   and all transformer blocks can receive real parameter updates.

On the one artificial full-shape diagnostic batch, forward and input-gradient
maximum absolute errors are zero on both devices. Maximum all-token parameter-
gradient errors are 2.61e-8 CPU / 2.98e-8 CUDA. Sampled MLP weight-gradient cosine
is 0.9904 / 0.9912, with relative errors 0.1405 / 0.1336. These are unit-test
diagnostics on random weights, not task-level gradient quality, significance,
learning efficacy or evidence that the estimator is accurate on trained models.

The full artificial CPU benchmark pipeline and its independent audit passed:
18 smoke trials, 54 retained artificial updates, 18 discarded warmups. Completed
resume preserves all files; tampering is rejected. These tiny CPU timings cannot
satisfy the GPU cost gate. No real checkpoint or data was accessed.

## Immediate next action

Run the frozen artificial operation-cost benchmark in
`protocol/sampled_weight_cost/COST_PROTOCOL_v1.md`. It compares native exact
backpropagation, all-token custom backpropagation (overhead control) and the
quarter-patch weight-gradient candidate. Two artificial seeds, batches 256/200,
six counterbalanced rounds, 72 trials. Gate on median paired candidate/native
update time <=90% in every seed/batch condition. Charge all sampling/gathering,
forward, loss, backward and optimizer work. Keep first-update allocation and
setup/warmup costs visible separately.

Run `scripts/run_sampled_weight_cost.cmd`, then review `results/sampled_weight_cost_v1`.
The full-shape GPU benchmark has not run during preparation. No real quality
study is frozen or launched. If this gate fails, park this fixed implementation;
do not tune its fraction from these timings. A pass only permits designing a
bounded gradient-quality/learning diagnostic and ultimately a complete adaptation
comparison with full and head-then-full controls at the existing quality target.
Cost and quality must both succeed before considering independent confirmation.
Publication remains on hold.
