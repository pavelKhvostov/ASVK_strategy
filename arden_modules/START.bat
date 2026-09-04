@echo off
chcp 65001 >nul
title Arden Live - сигналы стратегии Арденского
cd /d "%~dp0"

set "PY=%~dp0..\python\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo   ============================================================
echo     Стратегия Арденского - живые сигналы
echo   ============================================================
echo.
"%PY%" arden_live.py %*
echo.
echo   Раннер остановлен. Нажми любую клавишу.
pause >nul
