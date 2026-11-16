@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Artificial full-backpropagation backend screen. FP32 and strict determinism.
echo Stop other GPU work. Unsupported backends are recorded and skipped.
"%PYTHON%" -B scripts\test_efficient_backprop.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_efficient_backprop.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\efficient_backprop.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_efficient_backprop.py results\efficient_backprop_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\efficient_backprop_v1 for review.
