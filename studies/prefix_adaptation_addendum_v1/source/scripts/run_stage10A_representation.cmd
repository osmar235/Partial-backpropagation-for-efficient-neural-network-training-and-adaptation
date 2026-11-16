@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage10A: coarse/detail representation screen, five arms x two seeds x 100 epochs.
echo Full backpropagation in every arm; fixed targets; strict deterministic execution.
"%PYTHON%" -B scripts\test_stage10_representation.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage10_representation.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage10_representation.py --smoke --smoke-tag final_cpu
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage10_representation.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage10A_representation_v1 for review.
