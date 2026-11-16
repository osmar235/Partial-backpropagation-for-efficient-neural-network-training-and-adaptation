@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Competitive attention screen: artificial inputs, four sessions, no real-data training.
echo Stop other GPU work and keep the computer awake.
"%PYTHON%" -B scripts\attention_reference.py --run-all
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\review_attention_reference.py results\attention_reference_v1
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\attention_reference_v1 for review.
