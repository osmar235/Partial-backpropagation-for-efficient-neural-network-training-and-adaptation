@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo QKV real-data confirmation: 32 matched five-epoch trajectories.
echo Stop other GPU work. Keep the computer awake and allow several GB of disk space.
"%PYTHON%" -B scripts\qkv_training.py --run-all
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_qkv_training.py results\qkv_training_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\qkv_training_v1 for review.
