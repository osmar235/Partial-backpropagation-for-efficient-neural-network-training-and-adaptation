@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -B scripts\test_stage8_vit.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage8_workflow.py smoke
if errorlevel 1 exit /b 1
echo STAGE 8 SANITY COMPLETE. Artificial data only; no scientific inference.
