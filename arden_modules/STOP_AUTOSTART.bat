@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%~dp0..\python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" arden_live.py --uninstall
pause
