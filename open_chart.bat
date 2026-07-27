@echo off
rem ASVK Chart launcher - portable, nothing installed outside G:\ASVK.
rem Starts the local chart server (if not already running) and opens it in
rem Chrome (falls back to the default browser if Chrome isn't found).

setlocal
set "ASVK=%~dp0"
set "PORT=8080"
set "URL=http://127.0.0.1:%PORT%/"
set "LOG=%ASVK%logs\dashboard.log"

if not exist "%ASVK%logs" mkdir "%ASVK%logs"

rem --- already running? ---
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%URL%' -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo Starting ASVK Chart server on %URL% ...
    start "ASVK Chart Server" /min cmd /c ""%ASVK%python\python.exe" -m uvicorn server:app --host 127.0.0.1 --port %PORT% --app-dir "%ASVK%dashboard" >> "%LOG%" 2>&1"
    ping -n 3 127.0.0.1 >nul
) else (
    echo ASVK Chart server already running.
)

rem --- open in Chrome, else default browser ---
set "CHROME1=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
set "CHROME2=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
set "CHROME3=%LocalAppData%\Google\Chrome\Application\chrome.exe"

if exist "%CHROME1%" (
    start "" "%CHROME1%" --new-window "%URL%"
) else if exist "%CHROME2%" (
    start "" "%CHROME2%" --new-window "%URL%"
) else if exist "%CHROME3%" (
    start "" "%CHROME3%" --new-window "%URL%"
) else (
    start "" "%URL%"
)

endlocal
