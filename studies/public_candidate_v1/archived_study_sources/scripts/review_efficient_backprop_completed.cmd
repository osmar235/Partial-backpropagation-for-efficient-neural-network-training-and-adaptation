@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
"%PYTHON%" -B scripts\review_efficient_backprop_completed.py results\efficient_backprop_v1
if errorlevel 1 exit /b 1
echo Completed results audited. No experiment was rerun.
