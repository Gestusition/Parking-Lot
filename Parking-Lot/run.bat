@echo off
echo Starting Smart Parking System...
cd /d "%~dp0"
call venv\Scripts\activate
start /high /wait python main.py
pause
