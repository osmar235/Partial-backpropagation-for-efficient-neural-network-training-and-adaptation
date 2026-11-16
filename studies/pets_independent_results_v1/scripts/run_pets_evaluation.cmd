@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Oxford Pet independent evaluation: 24 seeds, three BF16 arms.
echo All training completes before official-test access. Close other GPU jobs.
"%PYTHON%" -B scripts\pets_evaluation.py run --device cuda
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\pets_independent_evaluation_v1 for review.
exit /b 0
