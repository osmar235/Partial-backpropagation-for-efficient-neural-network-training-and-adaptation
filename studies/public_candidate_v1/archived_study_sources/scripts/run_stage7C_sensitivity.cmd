@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 7C: EMA, prefix, and forced-full sensitivity. Seeds 57-62.
for %%S in (57 58 59 60 61 62) do (
  "%PYTHON%" -B scripts\stage7_review_experiments.py --suite sensitivity --dataset cifar10 --data-dir data --train-subset 50000 --eval-subset 10000 --batch-size 256 --epochs 30 --schedule-epochs 30 --seed %%S --device cuda --optimizer adam --lr 0.0003 --rho 0.95 --warmup-steps 196 --eval-batch-size 512 --log-steps --rotate-order --rotation-start 57 --out results\stage7C_sensitivity
  if errorlevel 1 exit /b 1
)
"%PYTHON%" -B scripts\stage7_summarize.py --root results\stage7C_sensitivity --reference lgpb --expected-seeds 57 58 59 60 61 62 --expected-variants lgpb rho_090 rho_099 no_forced_full prefix_stem prefix_stem_layer12
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage7_step_timing_summary.py --root results\stage7C_sensitivity
if errorlevel 1 exit /b 1
echo STAGE 7C COMPLETE.

