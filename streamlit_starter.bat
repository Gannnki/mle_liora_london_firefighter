@echo off
setlocal
cd /d "%~dp0"
uv sync --locked
if errorlevel 1 (
    echo [ERROR] Could not sync dependencies. Install uv first.
    pause
    exit /b 1
)
start "LFB API" cmd /k "uv run --locked uvicorn src.display_streamlit.api:app --host 127.0.0.1 --port 8000"
uv run --locked python scripts/run_ui.py
pause
