@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage11A: width expansion and cost feasibility at saved Stage10 checkpoints.
echo Short diagnostic only. Stop other GPU work before starting.
"%PYTHON%" -B scripts\test_stage11_growth.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage11_growth.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage11_growth_feasibility.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage11A_growth_feasibility_v1 for review.
