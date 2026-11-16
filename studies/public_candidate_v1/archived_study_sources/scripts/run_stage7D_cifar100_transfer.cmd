@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 7D: CIFAR-100 ResNet-18-GroupNorm transfer. Seeds 63-68.
echo Frozen controller unchanged; this is a new external-dataset test.
for %%S in (63 64 65 66 67 68) do (
  "%PYTHON%" -B scripts\stage7_review_experiments.py --suite transfer --dataset cifar100 --data-dir data --train-subset 50000 --eval-subset 10000 --batch-size 256 --epochs 30 --schedule-epochs 30 --seed %%S --device cuda --optimizer adam --lr 0.0003 --rho 0.95 --warmup-steps 196 --eval-batch-size 512 --log-steps --rotate-order --rotation-start 63 --out results\stage7D_cifar100_transfer
  if errorlevel 1 exit /b 1
)
"%PYTHON%" -B scripts\stage7_summarize.py --root results\stage7D_cifar100_transfer --reference adam --expected-seeds 63 64 65 66 67 68 --expected-variants adam lgpb
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage7_step_timing_summary.py --root results\stage7D_cifar100_transfer
if errorlevel 1 exit /b 1
echo STAGE 7D COMPLETE.

