# Run the optimized prefix/suffix cost screen

Prepared 2026-10-28. Full scientific run is pending. Close other GPU workloads.
From Windows CMD, using the existing project environment:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_optimized_prefix_cost.cmd
```

Return `results\optimized_prefix_cost_v1` when COMPLETE appears. The launcher
runs the audit automatically. A completed rerun verifies saved results without
retraining; an interrupted session is retained and retried in full.

Two sources, four timing sessions, 14 arms per source/session, 96 measured updates
per arm. Test cuts after blocks 1/3/5/6, suffix-only and refresh intervals 4/16,
against BF16/automatic full backpropagation. Full and routed-full controls must
match exactly. Final layer norm and classifier train even at cut 6.

This measures training-block costs, including augmentation and real refresh/routing
work. It does not test accuracy or measure complete adaptation speedup. Keep the
existing environment: the runner rejects changes to frozen code, data, checkpoints
and environment metadata. No packages need to be installed.

Protocol: `protocol/optimized_prefix_cost/COST_PROTOCOL_v1.md`.
