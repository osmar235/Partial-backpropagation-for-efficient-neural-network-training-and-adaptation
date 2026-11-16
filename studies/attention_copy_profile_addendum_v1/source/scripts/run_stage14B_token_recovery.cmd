@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage14B: five fixed recovery epochs; full, spatial and similarity; two seeds.
echo Stop other GPU work. Frozen source checkpoints stay read only.
"%PYTHON%" -B scripts\test_stage14_recovery.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage14_recovery.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage14_recovery.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage14B_token_recovery_v1 for review.
