@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem One-step local startup: setup venv + run UI.
set "PORT=%~1"
if not defined PORT set "PORT=8501"

call setup_env.bat
if errorlevel 1 exit /b 1

call run_ui.bat %PORT%
if errorlevel 1 exit /b 1

endlocal
exit /b 0
