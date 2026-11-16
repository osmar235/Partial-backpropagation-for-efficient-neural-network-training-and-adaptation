@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Artificial QKV layout screen. Full gradients, FP32, same math attention.
echo Stop other GPU work. Only a numerically validated layout is timed.
"%PYTHON%" -B scripts\test_qkv_layout.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_qkv_layout.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\qkv_layout.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_qkv_layout.py results\qkv_layout_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\qkv_layout_v1 for review.
