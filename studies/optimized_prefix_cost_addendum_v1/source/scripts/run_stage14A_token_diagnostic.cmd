@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage14A: midpoint token compression; no optimizer updates.
echo Stop other GPU work. Existing checkpoints are read only.
"%PYTHON%" -B scripts\test_stage14_tokens.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage14_tokens.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage14_diagnostic.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage14A_token_diagnostic_v1 for review.
