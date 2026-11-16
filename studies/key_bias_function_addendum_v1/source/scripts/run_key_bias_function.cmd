@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Key-bias functional diagnostic. Keep other GPU jobs stopped.
echo Replays saved optimizer steps and evaluates fixed batches. No new gradients.
"%PYTHON%" -B scripts\key_bias_function.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_key_bias_function.py results\key_bias_function_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\key_bias_function_v1 for review.
