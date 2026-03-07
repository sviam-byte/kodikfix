@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Detect a working Python bootstrap command (prefer py -3, then python, then python3).
set "BOOTSTRAP_CMD="

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -c "import sys" >nul 2>nul
    if %errorlevel%==0 set "BOOTSTRAP_CMD=py -3"
)

if not defined BOOTSTRAP_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python -c "import sys" >nul 2>nul
        if %errorlevel%==0 set "BOOTSTRAP_CMD=python"
    )
)

if not defined BOOTSTRAP_CMD (
    where python3 >nul 2>nul
    if %errorlevel%==0 (
        python3 -c "import sys" >nul 2>nul
        if %errorlevel%==0 set "BOOTSTRAP_CMD=python3"
    )
)

if not defined BOOTSTRAP_CMD (
    echo [ERROR] Working Python interpreter not found in PATH.
    echo [HINT] Install official Python from python.org and make sure pip is available.
    pause
    exit /b 1
)

echo [INFO] Bootstrap interpreter: %BOOTSTRAP_CMD%

if not exist ".venv\Scripts\python.exe" (
    echo [STEP] Creating virtual environment...
    %BOOTSTRAP_CMD% -m venv .venv
    if errorlevel 1 (
        rem Fallback path for broken stdlib venv/ensurepip installations.
        echo [WARN] stdlib venv failed, trying virtualenv fallback...
        %BOOTSTRAP_CMD% -m pip install --upgrade pip virtualenv
        if errorlevel 1 (
            echo [ERROR] Failed to prepare virtualenv fallback.
            pause
            exit /b 1
        )
        %BOOTSTRAP_CMD% -m virtualenv .venv
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment.
            echo [HINT] Your Python installation may be incomplete.
            pause
            exit /b 1
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv was not created correctly.
    pause
    exit /b 1
)

echo [STEP] Upgrading pip/setuptools/wheel...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to install setuptools/wheel
    pause
    exit /b 1
)

if exist "requirements.txt" (
    echo [STEP] Installing requirements...
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
echo [OK] Environment is ready.
echo Use:
echo   run_ui.bat
echo   run_cli.bat metrics data.csv --src src --dst dst
pause
endlocal
