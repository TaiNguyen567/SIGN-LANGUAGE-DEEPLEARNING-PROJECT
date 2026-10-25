@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo [INFO] Dang mo camera. Tu vung: Xin loi, An, va tu An mung den Cam on.
    ".venv\Scripts\python.exe" app.py --device auto
) else (
    ".venv\Scripts\python.exe" app.py %*
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
