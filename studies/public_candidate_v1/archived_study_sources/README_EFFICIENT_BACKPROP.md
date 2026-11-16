# Efficient full-backpropagation test

First test: attention backends with full gradients, unchanged FP32 precision and
strict determinism. No sampled gradients or tensor-layout rewrite is included.
Unsupported backends are recorded and skipped; numerical failures block timing.
The six-block model, two seeds and batches 256/200 are tested on artificial data.

Close other GPU workloads. Run in CMD:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_efficient_backprop.cmd
```

Send `results\efficient_backprop_v1`, including its subfolders, when complete.
Rerun the same command after interruption; completed conditions are verified
and skipped. Preserve the folder even if no alternative backend qualifies.

The command checks CPU/CUDA correctness, verifies actual backend dispatch and
compares supported implementations with native math attention over six paired
rounds per condition. No silent fallback or precision change is allowed. Every
logit, gradient, parameter and AdamW moment is checked on a short trajectory
before timing. A practical cost result requires at least 10% median paired saving
in all four conditions. It does not establish real-data accuracy or novelty.

At most 96 timing trials (864 timed updates plus 288 discarded warmups), with
separate capability/correctness probes. Setup, first-update cost and memory are
reported too. Full details: `protocol/efficient_backprop/BACKEND_PROTOCOL_v1.md`.
Only small artificial CPU/CUDA checks are run during preparation; the full
screen is for the command above. No real dataset or checkpoint is used.

Preparation finding: the small GPU pipeline works, but efficient attention fails
the post-update parameter tolerance in an attention key-bias slice. Its logits,
gradients and AdamW moments pass their checks. Flash is unavailable in this build;
cuDNN attention is unsupported under strict determinism. These are recorded
outcomes, not hidden fallbacks. The full-shape test repeats all checks and will
skip timing if the parameter mismatch persists. No tolerance has been relaxed,
and no alternative backend is qualified by the preparation run.
