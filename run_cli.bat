@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" goto usage

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto run_cli

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=py"
    goto run_cli
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
    goto run_cli
)

echo [ERROR] Python not found.
echo Run setup_env.bat first or install Python.
pause
exit /b 1

:run_cli
"%PYTHON_EXE%" run_local.py cli %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] CLI finished with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:usage
echo Usage:
echo   run_cli.bat metrics data.csv --src src --dst dst
echo   run_cli.bat attack data.csv --family node --kind degree --frac 0.5
echo   run_cli.bat mixfrac --patient patient.csv --healthy hc1.csv hc2.csv
pause
exit /b 1
