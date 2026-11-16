@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Bounded optimized attention/copy profiling. Stop other GPU jobs.
echo This is a cost diagnostic, with all gradients retained; no accuracy qualification.
"%PYTHON%" -B scripts\attention_copy_profile.py --run-all
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\attention_copy_profile_v1 for review.
