@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "BOOTSTRAP_PY=py"
    goto bootstrap_ok
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "BOOTSTRAP_PY=python"
    goto bootstrap_ok
)

echo [ERROR] Python not found in PATH.
pause
exit /b 1

:bootstrap_ok
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    "%BOOTSTRAP_PY%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        pause
        exit /b 1
    )
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)

if exist "requirements.txt" (
    echo Installing requirements...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
) else (
    echo [WARN] requirements.txt not found, skipping install.
)

echo.
echo Environment is ready.
echo Use:
echo   run_ui.bat
echo   run_cli.bat metrics data.csv --src src --dst dst
pause
endlocal
