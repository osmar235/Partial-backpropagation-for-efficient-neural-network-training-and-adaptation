@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage11B: two fresh seeds, five arms, 100 epochs, one midpoint expansion.
echo Stop other GPU workloads. No automatic extension or parameter search.
"%PYTHON%" -B scripts\test_stage11_growth_pilot.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage11_growth_pilot.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage11_growth_pilot.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage11B_growth_pilot_v1 for review.
