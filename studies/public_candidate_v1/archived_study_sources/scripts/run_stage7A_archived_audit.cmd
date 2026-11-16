@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -B scripts\stage7_audit_existing.py --results-dir . --out results\stage7A_original_audit
if errorlevel 1 exit /b 1
echo AUDIT COMPLETE. No GPU experiment was run.

