@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -B scripts\test_stage9_boundary_diagnostic.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage9_boundary_diagnostic.py --smoke
if errorlevel 1 exit /b 1
echo ARTIFICIAL CPU WORKFLOW COMPLETE. No scientific performance evidence.
