@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 9C: six ViT boundaries, two shared 30-epoch development baselines.
echo Checkpoints 1,5,15,30; immediate update utility and measured microstep costs.
"%PYTHON%" -B scripts\test_stage9_boundary_diagnostic.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage9_boundary_diagnostic.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage9C_boundary_diagnostic_v1 for review.
