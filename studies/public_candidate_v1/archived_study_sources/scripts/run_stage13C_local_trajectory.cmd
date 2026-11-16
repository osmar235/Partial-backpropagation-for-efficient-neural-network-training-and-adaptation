@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo Stage13C: matched local/full trajectories with restored optimizer history.
echo Stop other GPU work. Eight fixed one-epoch trajectories; no automatic extension.
"%PYTHON%" -B scripts\test_stage13c_trajectory.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\test_stage13c_trajectory.py --cuda
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage13c_trajectory.py
if errorlevel 1 exit /b 1
echo COMPLETE. Send results\stage13C_local_trajectory_v1 for review.
