@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 8A: training-only ViT development; six runs if baseline readiness passes.
"%PYTHON%" -B scripts\stage8_workflow.py development
if errorlevel 1 exit /b 1
echo DEVELOPMENT COMPLETE. Send results\stage8A_vit_development for review before evaluation.
