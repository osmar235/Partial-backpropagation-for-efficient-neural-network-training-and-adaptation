# Optimized-execution feasibility — 2026-10-27

Native and upstream-reference attention, each with FP32/math, BF16/math and
BF16/automatic SDPA. The reference FP32/math arm is the required baseline.
FP32 math remains strict deterministic; automatic BF16 explicitly permits
nondeterministic optimized kernels. All model parameters and AdamW state remain
FP32. Actual backend dispatch, numerical checks and startup/steady clocks are
recorded separately. No dataset or trained checkpoint is loaded.

The local tiny CUDA Inductor probe failed with TritonMissing. No compiler or
environment changes were made; compiled performance is unavailable in this
screen. This is distinct from BF16, which the GPU supports.

Preparation is validated and frozen. Run in CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_optimized_execution.cmd
```

Return `results\optimized_execution_v1` with subfolders. Full eligibility gives
576 trials /19,008 post-restore measured updates, plus startup diagnostics,
warmups and numerical/profiling checks. Failed/unsupported arms are recorded
and not silently rescued. A completed rerun verifies and skips saved work;
retain interrupted attempts. Stop other GPU work during execution.

Numerical qualification disables dropout in every arm to isolate arithmetic
from mask-generation differences. Timings retain normal training dropout.
BF16 limits are prospective numerical sanity budgets, **not accuracy-equivalence
criteria**. Even a faster qualifying arm needs a separate real-data quality and
complete-workflow study. Existing FP32 tolerances and historical findings remain
unchanged. Detailed design: `protocol/optimized_execution/EXECUTION_PROTOCOL_v1.md`.

Preparation checks: three CPU tests passed (one CUDA-only test skipped), all
four CUDA tests passed, and both two-process tiny workflows audited. Each
workflow timed only the eligible reference FP32/math baseline. Tiny native
FP32 failed the post-Adam projection-bias tolerance; all tiny CUDA BF16 arms
failed the fixed post-update parameter sanity bound. No limit was relaxed.
These are small-model diagnostics; full-size eligibility is unknown. Separate
BF16 trial-function checks passed with normal dropout, without waiving the
study gates or producing scientific timing evidence. Completed-run skipping
was verified. Automatic BF16 used memory-efficient SDPA in the tiny probe.

Preparation record: `results/optimized_execution_preparation_v1/REPORT.md`.
Organized source capsule: `paper_support/optimized_execution_addendum_v1/`.
Current project master: v7.38; paper discussion memory: v27.
