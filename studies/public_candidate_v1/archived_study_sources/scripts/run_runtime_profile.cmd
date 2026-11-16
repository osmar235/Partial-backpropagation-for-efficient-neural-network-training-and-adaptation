@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Artificial runtime attribution only. No real dataset or checkpoint.
echo Stop other GPU work. 12 conditions, 336 measured updates plus 36 burn-in.
"%PYTHON%" -B scripts\test_runtime_profile.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_runtime_profile.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\runtime_profile.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_runtime_profile.py results\runtime_profile_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\runtime_profile_v1 for review.
