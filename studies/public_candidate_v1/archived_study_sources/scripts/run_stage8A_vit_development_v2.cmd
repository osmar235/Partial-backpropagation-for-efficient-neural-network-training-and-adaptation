@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Stage 8A v2: fixed LR 0.001, MixUp 0.2, 300 epochs, two baseline seeds.
echo Two LGPB calibrations run only if the unchanged 85%% mean and 83%% minimum pass.
"%PYTHON%" -B scripts\stage8_workflow_v2.py development
if errorlevel 1 exit /b 1
echo DEVELOPMENT COMPLETE. Send results\stage8A_vit_development_v2 for review before evaluation.
