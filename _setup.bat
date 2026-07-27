@echo off
chcp 65001 >nul
title ASVK Setup — проверка bundled-окружения
setlocal
set "ASVK=%~dp0"

echo Проверка портативного Python и зависимостей (ничего не устанавливается —
echo всё уже лежит в %ASVK%python\)...
echo.

if not exist "%ASVK%python\python.exe" (
    echo [FAIL] %ASVK%python\python.exe не найден.
    echo        Папка python\ должна быть скопирована вместе с ASVK целиком.
    pause
    exit /b 1
)

"%ASVK%python\python.exe" -c "import importlib.metadata as im; import requests, rich, winotify, numpy, pandas, fastapi, uvicorn, numba, scipy, joblib, pyarrow, websockets; print('[OK] все зависимости на месте: requests', im.version('requests'), '| rich', im.version('rich'), '| pandas', im.version('pandas'))"
if errorlevel 1 (
    echo.
    echo [FAIL] Bundled python\ есть, но не хватает какого-то пакета — папка
    echo        python\ скопирована не полностью или повреждена.
    pause
    exit /b 1
)

echo.
echo === Всё на месте. Можно запускать ASVK.bat ===
pause
