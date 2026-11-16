@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo QKV confirmation: eight fresh processes, longer paired timing blocks.
echo Stop other GPU work and keep the computer awake. No real-data training.
"%PYTHON%" -B scripts\qkv_confirmation.py --run-all
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_qkv_confirmation.py results\qkv_confirmation_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\qkv_confirmation_v1 for review.
