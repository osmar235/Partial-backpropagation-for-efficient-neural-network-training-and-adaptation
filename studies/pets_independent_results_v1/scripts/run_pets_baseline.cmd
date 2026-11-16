@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Oxford-IIIT Pet baseline qualification. Official test evaluation is disabled.
echo Close other GPU jobs. Dataset download and caching occur only if needed.
"%PYTHON%" -B scripts\pets_transfer.py prepare
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\pets_transfer.py baseline --device cuda
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\pets_baseline_v1 for review. No test phase was run.
