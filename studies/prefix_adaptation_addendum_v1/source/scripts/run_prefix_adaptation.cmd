@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Full, frozen cut3, periodic cut3/r4: quality and COMPLETE adaptation time.
echo Stop other GPU jobs before running. The final audit performs inference.
"%PYTHON%" -B scripts\prefix_adaptation.py --run-all
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\prefix_adaptation_v1 for review.
