@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto run_ui

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=py"
    goto run_ui
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
    goto run_ui
)

echo [ERROR] Python not found.
echo Run setup_env.bat first or install Python.
pause
exit /b 1

:run_ui
echo Starting Kodik Lab UI...
"%PYTHON_EXE%" run_local.py ui
if errorlevel 1 (
    echo.
    echo [ERROR] UI failed to start.
    pause
    exit /b 1
)

endlocal
