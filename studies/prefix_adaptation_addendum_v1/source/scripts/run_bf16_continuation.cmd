@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Matched BF16 quality and complete adaptation cost. Stop other GPU jobs.
"%PYTHON%" -B scripts\bf16_continuation.py --run-all
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\bf16_continuation_v1 for review.
