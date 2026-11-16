@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
echo Recover official Oxford archives with retries and checksum verification.
echo Partial files are preserved when a server cannot resume. No training settings change.
"%PYTHON%" -B scripts\download_pets_archives.py
if errorlevel 1 exit /b 1
call scripts\run_pets_baseline.cmd
exit /b %errorlevel%
