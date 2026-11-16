@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage15A: 10 runs, 5 epochs each. Full, head, head-then-full, two side controls.
echo Stop other GPU work. Source checkpoints and previous evidence stay read only.
"%PYTHON%" -B scripts\test_stage15_side_adaptation.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage15_side_adaptation.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage15_side_adaptation.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage15A_side_adaptation_v1 for review.
