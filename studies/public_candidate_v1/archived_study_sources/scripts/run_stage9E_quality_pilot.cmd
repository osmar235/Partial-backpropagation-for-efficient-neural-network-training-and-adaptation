@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage9E: five arms x two development seeds x 100 epochs, cut4, prefix freezing.
echo Fixed targets and matched budgets. No automatic extension or test evaluation.
"%PYTHON%" -B scripts\test_stage9_quality_pilot.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage9_quality_pilot.py --smoke --smoke-tag final_cpu
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage9_quality_pilot.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage9E_quality_pilot_v1 for review.
