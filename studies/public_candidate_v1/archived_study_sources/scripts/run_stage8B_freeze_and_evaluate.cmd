@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 8B: freeze completed development, then run 24 new paired ViT evaluations.
"%PYTHON%" -B scripts\stage8_workflow.py freeze-evaluate
if errorlevel 1 exit /b 1
echo STAGE 8B COMPLETE. Send results\stage8B_vit_evaluation and protocol\stage8\FROZEN_EVALUATION.json.
