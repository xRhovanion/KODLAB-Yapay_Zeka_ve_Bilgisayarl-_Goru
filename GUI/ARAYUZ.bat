@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Kurulum icin docs\CALISTIRMA.md dosyasina bakin.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "GUI\GUI.py"
if errorlevel 1 pause
