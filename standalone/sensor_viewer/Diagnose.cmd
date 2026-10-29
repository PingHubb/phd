@echo off
cd /d "%~dp0"
"%~dp0runtime\python.exe" -B "%~dp0app\run.py" --diagnostics
pause
