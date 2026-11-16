@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Optimized prefix/suffix COST ONLY. Stop other GPU jobs before running.
"%PYTHON%" -B scripts\optimized_prefix_cost.py --run-all
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\optimized_prefix_cost_v1 for review.
