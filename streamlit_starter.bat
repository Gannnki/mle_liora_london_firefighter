@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not find .venv\Scripts\python.exe
    echo Please create or restore the project virtual environment first.
    pause
    exit /b 1
)

echo Checking Streamlit installation...
.venv\Scripts\python.exe -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit is not installed in .venv. Installing Streamlit app requirements...
    .venv\Scripts\python.exe -m pip install -r src\display_streamlit\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install Streamlit requirements.
        pause
        exit /b 1
    )
)

cd /d "%~dp0src\display_streamlit"

echo Starting London Fire Brigade Streamlit app...
echo Open: http://localhost:8501
"%~dp0.venv\Scripts\python.exe" -m streamlit run streamlit_app.py

pause
