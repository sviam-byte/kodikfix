@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PORT=%~1"
if not defined PORT set "PORT=8501"

if not exist ".venv\Scripts\python.exe" if not exist ".venv\bin\python" (
    call setup_env.bat
    if errorlevel 1 exit /b 1
)

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%CD%\.venv\bin\python"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python from .venv not found after setup.
    pause
    exit /b 1
)

if not exist "logs" mkdir logs >nul 2>nul

echo [STEP] Starting Kodik Lab UI on http://127.0.0.1:%PORT% ...
start "" "http://127.0.0.1:%PORT%"

"%PYTHON_EXE%" run_local.py ui -- --server.port=%PORT% 1>logs\ui_stdout.log 2>logs\ui_stderr.log
if errorlevel 1 (
    echo.
    echo [ERROR] UI failed to start.
    echo [HINT] See logs\ui_stderr.log
    pause
    exit /b 1
)

endlocal
