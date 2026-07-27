@echo off
chcp 65001 >nul
title ASVK Data Collector
cd /d %~dp0

python\python.exe asvk.py
if errorlevel 1 (
    echo.
    echo === ASVK exited with error ===
    echo Check logs\asvk.log
    pause
) else (
    echo.
    echo === ASVK stopped clean ===
    pause
)
