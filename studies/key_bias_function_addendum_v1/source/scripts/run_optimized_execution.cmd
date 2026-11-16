@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Optimized execution feasibility: artificial inputs, four fresh processes.
echo Stop other GPU work and keep the computer awake.
echo Compilation capability is recorded separately; no compiled timing is substituted.
"%PYTHON%" -B scripts\optimized_execution.py --run-all
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_optimized_execution.py results\optimized_execution_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\optimized_execution_v1 for review.
