@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 7B: new CIFAR-10 mechanism controls, six seeds 51-56.
for %%S in (51 52 53 54 55 56) do (
  "%PYTHON%" -B scripts\stage7_review_experiments.py --suite core --dataset cifar10 --data-dir data --train-subset 50000 --eval-subset 10000 --batch-size 256 --epochs 30 --schedule-epochs 30 --seed %%S --device cuda --optimizer adam --lr 0.0003 --rho 0.95 --matched-q 0.35990173847316703 --warmup-steps 196 --eval-batch-size 512 --log-steps --rotate-order --rotation-start 51 --out results\stage7B_core_ablation
  if errorlevel 1 exit /b 1
)
"%PYTHON%" -B scripts\stage7_summarize.py --root results\stage7B_core_ablation --reference lgpb --additional-references adam --expected-seeds 51 52 53 54 55 56 --expected-variants adam lgpb random_matched periodic_matched freeze_prefix replay_prefix
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage7_step_timing_summary.py --root results\stage7B_core_ablation
if errorlevel 1 exit /b 1
echo STAGE 7B COMPLETE. Do not treat these post-confirmation results as part of Stage 6B preregistration.

