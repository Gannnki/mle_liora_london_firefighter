@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not find .venv\Scripts\python.exe
    echo Please create or restore the project virtual environment first.
    pause
    exit /b 1
)

set RUN_NAME=xgboost-pipeline-%date:~-4%%date:~3,2%%date:~0,2%-%time:~0,2%%time:~3,2%%time:~6,2%
set RUN_NAME=%RUN_NAME: =0%

echo Starting XGBoost pipeline...
echo Run name: %RUN_NAME%
echo.

.venv\Scripts\python.exe src\pipeline.py --run-name "%RUN_NAME%"

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed.
    pause
    exit /b 1
)

echo.
echo Pipeline finished successfully.
pause
