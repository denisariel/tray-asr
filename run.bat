@echo off
REM Speech Recognition Tray App - Windows launcher

REM Get the directory where this script is located
cd /d "%~dp0"

REM Create venv if it doesn't exist or is broken
if not exist ".venv\Scripts\python.exe" (
    echo Setting up virtual environment with uv...
    uv venv
    uv pip install -r requirements.txt
    echo Setup complete!
)

REM Run the app
.venv\Scripts\python.exe main.py
