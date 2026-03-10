@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem One-step local startup: setup venv + run UI.
set "PORT=%~1"
rem Keep default in sync with run_ui.bat.
if not defined PORT set "PORT=8502"

call setup_env.bat --no-pause
if errorlevel 1 exit /b 1

call run_ui.bat %PORT%
if errorlevel 1 exit /b 1

endlocal
exit /b 0
