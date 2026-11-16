@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Oxford Pet selective development: six seeds, three BF16 arms, trainval only.
echo Close other GPU jobs. Existing cache and weights will be verified.
"%PYTHON%" -B scripts\pets_selective.py development --device cuda
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\pets_selective_development_v1 for review.
exit /b 0
