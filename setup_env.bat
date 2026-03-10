@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Optional flag for caller scripts: skip final pause in non-interactive flows.
set "NO_PAUSE="
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

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
    if not defined NO_PAUSE pause
    exit /b 1
)

echo [INFO] Bootstrap interpreter: %BOOTSTRAP_CMD%
%BOOTSTRAP_CMD% tools\bootstrap_env.py
if errorlevel 1 (
    echo.
    echo [ERROR] Environment setup failed.
    if not defined NO_PAUSE pause
    exit /b 1
)

if not defined NO_PAUSE pause
endlocal
