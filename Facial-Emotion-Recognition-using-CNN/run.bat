@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE="%~dp0.venv\Scripts\python.exe"

if not exist %PYTHON_EXE% (
    echo [ERROR] Virtual environment python not found at %PYTHON_EXE%
    pause
    exit /b 1
)

echo Starting Facial Emotion Recognition...
%PYTHON_EXE% "%~dp0main.py" %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo Program exited with error code %ERRORLEVEL%.
    pause
)
