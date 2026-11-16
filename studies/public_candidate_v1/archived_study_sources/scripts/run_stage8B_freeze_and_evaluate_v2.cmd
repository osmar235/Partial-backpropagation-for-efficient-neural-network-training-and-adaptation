@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 8B v2: freeze reviewed development and run all 24 paired evaluations.
"%PYTHON%" -B scripts\stage8_workflow_v2.py freeze-evaluate
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage8B_vit_evaluation_v2 and protocol\stage8\FROZEN_EVALUATION_v2.json.
