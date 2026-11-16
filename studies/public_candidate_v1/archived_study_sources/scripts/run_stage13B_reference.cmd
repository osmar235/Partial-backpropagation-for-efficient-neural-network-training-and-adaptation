@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage13B: qualify full backpropagation before testing local learning.
echo Stop other GPU work. This runs four fixed one-epoch continuations.
"%PYTHON%" -B scripts\test_stage13b_reference.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage13b_reference.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage13b_reference.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage13B_reference_v1 for review.
