@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Matched-state optimizer diagnostic. Keep other GPU jobs stopped.
echo This records raw tensors; allow several GB of disk space.
"%PYTHON%" -B scripts\optimizer_sensitivity.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_optimizer_sensitivity.py results\optimizer_sensitivity_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\optimizer_sensitivity_v1 for review.
