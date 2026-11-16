@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage12A: pretrained adaptation, two seeds, five arms, 12 epochs.
echo Stop other GPU workloads. Official ImageNet weights are already prepared.
"%PYTHON%" -B scripts\test_stage12_adaptation.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage12_adaptation.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage12_adaptation.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage12A_adaptation_v1 for review.
