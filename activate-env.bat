@echo off
REM Activate Python virtual environment
cd /d "%~dp0"
call .venv\Scripts\activate.bat
cmd /k
