# Competitive attention reference — first controlled screen

Compare native attention, the unchanged QKV adapter, and the exact attention
class from pinned timm v0.9.2 in the same full model. FP32, strict determinism,
math SDPA, identical weights/inputs/optimizer/stochastic streams. No dataset
or trained checkpoint is loaded. This is an artificial operation-cost study.
The reference wrapper restores native output strides so downstream dropout
uses the same mask mapping; that compatibility copy is included in its timing.

From Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_attention_reference.cmd
```

Return `results\attention_reference_v1` with its subfolders. The full run is
frozen before release of these instructions. A normal rerun verifies and skips
completed conditions. Interrupted attempts remain preserved and are reported.
Do not edit frozen code/specs to clear a gate or delete unsuccessful attempts.

Four fresh processes, two seeds, two batch sizes, six order-balanced rounds.
At most 288 trials / 9,504 measured updates plus warmups and numerical checks.
Actual cost depends on eligible arms and the machine; no runtime promise.
Stop other GPU work. Full-shape numerical failures are recorded outcomes;
those arms are excluded from timing rather than silently changing tolerances.

This screen asks whether an established packed-QKV implementation already
obtains the benefit or improves on it. It does not benchmark a whole timm model
or claim superiority over the latest optimized stack. Mixed precision and
compilation remain separate follow-ups. The earlier QKV results and failed
10% discovery gate stay unchanged. See
`protocol/attention_reference/REFERENCE_PROTOCOL_v1.md` for the full design.
