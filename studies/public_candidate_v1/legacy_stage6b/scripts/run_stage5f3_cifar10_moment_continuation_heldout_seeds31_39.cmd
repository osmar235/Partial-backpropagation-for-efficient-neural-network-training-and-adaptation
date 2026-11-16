@echo off
setlocal EnableDelayedExpansion
set "REPO=%~dp0.."
if not defined PYTHON set "PYTHON=python"
cd /d "%REPO%"
for /L %%S in (31,1,39) do (
  set ORDER=baseline_first
  set /A REM=%%S%%2
  if !REM!==0 set ORDER=candidate_first
  echo Running frozen moment-continuation held-out seed %%S with !ORDER!
  "%PYTHON%" -B ".\src\agrls_stage5d4_cifar10_heldout.py" --data-dir data --train-subset 50000 --eval-subset 10000 --batch-size 256 --baseline-updates 500 --equal-wall-updates 540 --seed %%S --device cuda --adam-lr 0.0003 --adan-lr 0.0003 --ema-decay 0.95 --candidate-mechanism loss_gated_moment_continuation_adam --algorithm-order !ORDER! --eval-batch-size 512 --out ".\results\stage5f3\stage5f3_cifar10_moment_continuation_heldout_seed%%S"
  if errorlevel 1 exit /b !errorlevel!
)
"%PYTHON%" -B ".\src\agrls_stage5d4_summarize_heldout.py" --root ".\results\stage5f3" --run-prefix "stage5f3_cifar10_moment_continuation_heldout_seed" --seeds 31 32 33 34 35 36 37 38 39 --out ".\results\stage5f3\stage5f3_cifar10_moment_continuation_heldout_summary"
endlocal
