@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 calistir.py
) else (
    echo Once README dosyasindaki kurulum adimlarini tamamlayin.
)
echo.
pause
