@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Oxford recovery v2: progressing downloads may take longer than one hour.
echo No overall transfer deadline. Partial files and verified archives are preserved.
"%PYTHON%" -B scripts\download_pets_archives_v2.py
if errorlevel 1 exit /b 1
call scripts\run_pets_baseline.cmd
exit /b %errorlevel%
