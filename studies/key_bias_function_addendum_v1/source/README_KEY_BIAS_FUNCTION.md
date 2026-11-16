# Key-bias functional-impact diagnostic

This diagnostic asks how much of the saved update discrepancy changes model
outputs and loss. It compares the full update, a key-bias-only swap, and the full
update with key bias restored to the FP32 reference. All historical results and
gates remain unchanged. No long-run training quality claim follows from this test.

Run in Windows CMD from the original workspace with the existing environment:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_key_bias_function.cmd
```

Keep other GPU workloads stopped. No installations or downloads are required.
The frozen sources, environment, previous optimizer diagnostic, raw saved
gradients, archived checkpoints and CIFAR training files must remain available.
This launcher verifies them and aborts if they differ. Completed conditions are
verified and skipped on restart; interrupted attempts are retained.

Two trained sources, two anchors and two optimizer histories per anchor give
eight source conditions. The native FP32 control and four BF16 gradient origins
give 40 component cases. Every state is evaluated with the same reference
attention implementation under FP32/math, BF16/math and BF16/automatic execution.
Use the original fixed training batch and a disjoint second training-only batch.
At full availability this is 840 forward evaluations and 48 saved-gradient AdamW
replays. No new backward pass or learning trajectory is computed. The auditor
repeats the 48 isolated optimizer replays and reconstructs stored-logit statistics
without model forwards or backwards. Replays are inexpensive state checks, not
additional training. CPU smoke skips BF16 policies explicitly.

Outputs: `results/key_bias_function_v1/`, including per-condition saved logits,
input provenance, component hashes, dispatch, repeated-reference checks and
recomputed summaries. Send that folder for review, or reply “done” in this chat.
Uniform FP64 rescoring separates within-policy weight effects from differences
between evaluation precisions. Explicit FP32 scoring from the same logits is
checked against FP64; original autocast execution loss is retained separately.
Tiny validation found these loss paths can differ in this installed runtime.
Ratios are descriptive and may exceed one.
An integrity PASS does not mean functional equivalence or a quality PASS.

Prospective design: `protocol/key_bias_function/FUNCTIONAL_PROTOCOL_v1.md`.
Freeze: `protocol/key_bias_function/FROZEN_FUNCTIONAL_v1.json`.
Preparation evidence: `results/key_bias_function_preparation_v1/`.
Paper-support source capsule: `paper_support/key_bias_function_addendum_v1/`.
The capsule retains local absolute provenance; portable release packaging is
still pending. No full functional diagnostic is run during preparation.
