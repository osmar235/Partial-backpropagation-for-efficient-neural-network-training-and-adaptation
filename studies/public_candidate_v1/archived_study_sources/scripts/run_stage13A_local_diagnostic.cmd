@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage13A: short three-block local-loss mechanics, cost and utility diagnostic.
echo Stop other GPU work. Existing Stage12 source checkpoints remain unchanged.
"%PYTHON%" -B scripts\test_stage13_local.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage13_local.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage13_diagnostic.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage13A_local_diagnostic_v1 for review.
