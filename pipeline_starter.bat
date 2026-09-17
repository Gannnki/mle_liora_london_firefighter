@echo off
setlocal
cd /d "%~dp0"
uv run --locked python src\pipeline.py %*
if errorlevel 1 (
    echo [ERROR] Pipeline failed. Make sure uv is installed and input data exists.
)
pause
