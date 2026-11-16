@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage9D: cost-only batch256 check at existing checkpoints. No retraining.
"%PYTHON%" -B scripts\stage9_deployment_cost.py --smoke
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage9_deployment_cost.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage9D_deployment_cost_v1 for review.
