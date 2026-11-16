# Targeted QKV layout test

One change: split the packed QKV tensor with `unbind` instead of three indexed
views. The goal is to reduce backward scatter/add work. Keep the existing math
attention backend, full gradients, FP32, dropout and strict determinism.

Close other GPU work and run in CMD:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_qkv_layout.cmd
```

Send `results\qkv_layout_v1`, including subfolders, when complete.
The command runs correctness checks, the frozen artificial cost screen and a
read-only audit. The audit accepts relative paths. Rerunning the same command
validates/skips completed conditions and preserves incomplete attempts.

Two seeds, two batch sizes, six paired rounds, at most 48 timing trials.
The candidate must pass numerical and split-mechanism checks before timing and
save at least 10% median paired time in every condition. No relaxed key-bias
tolerance, gradient approximation or backend change is included. Setup, first
update and memory are also reported. No real training or automatic follow-up.

Protocol: `protocol/qkv_layout/LAYOUT_PROTOCOL_v1.md`. Small CPU/CUDA preparation
checks do not establish the full workload's speed or real-data quality.
