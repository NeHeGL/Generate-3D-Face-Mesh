@echo off
setlocal
title Generate 3D Mesh
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up Generate 3D Mesh for the first time...
    py -3 -m venv .venv || (echo Python 3 is needed: https://www.python.org/downloads/ & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)
rem pythonw runs without a console window; "start" hands over to it and this window closes straight away. Errors go to mesh-app.log.
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" run.py app
) else (
    ".venv\Scripts\python.exe" run.py app
)
exit /b 0
