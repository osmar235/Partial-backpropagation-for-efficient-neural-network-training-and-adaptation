@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Artificial sampled-weight-gradient cost gate. No real dataset or checkpoint.
echo Stop other GPU work. 72 trials, three implementations, two batch sizes.
"%PYTHON%" -B scripts\test_sampled_weight_gradient.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_sampled_weight_gradient.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\sampled_weight_cost.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\sampled_weight_cost_v1 for review.
