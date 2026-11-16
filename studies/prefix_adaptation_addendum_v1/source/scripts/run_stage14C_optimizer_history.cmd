@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage14C: 12 runs, 5 epochs each. Full/spatial/similarity x restored/fresh AdamW.
echo Stop other GPU work. Frozen source checkpoints stay read only.
"%PYTHON%" -B scripts\test_stage14_optimizer_history.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage14_optimizer_history.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage14_optimizer_history.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage14C_optimizer_history_v1 for review.
