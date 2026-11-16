@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo OPTIONAL Stage 7E: CIFAR-10 AdamW with 5e-4 decoupled weight decay, seeds 69-74.
echo This tests a NEW optimizer/recipe. It does not change Stage 6B.
for %%S in (69 70 71 72 73 74) do (
  "%PYTHON%" -B scripts\stage7_review_experiments.py --suite transfer --dataset cifar10 --data-dir data --train-subset 50000 --eval-subset 10000 --batch-size 256 --epochs 30 --schedule-epochs 30 --seed %%S --device cuda --optimizer adamw --weight-decay 0.0005 --lr 0.0003 --rho 0.95 --warmup-steps 196 --eval-batch-size 512 --log-steps --rotate-order --rotation-start 69 --out results\stage7E_adamw_optional
  if errorlevel 1 exit /b 1
)
"%PYTHON%" -B scripts\stage7_summarize.py --root results\stage7E_adamw_optional --reference adam --expected-seeds 69 70 71 72 73 74 --expected-variants adam lgpb
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage7_step_timing_summary.py --root results\stage7E_adamw_optional
if errorlevel 1 exit /b 1
echo STAGE 7E COMPLETE.

