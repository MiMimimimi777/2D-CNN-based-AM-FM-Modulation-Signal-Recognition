@echo off
chcp 65001 >nul
title AM-FM Modulation Recognition System

set "FLASK_APP=app.py"
set "FLASK_HOST=127.0.0.1"
set "FLASK_PORT=5000"

echo.
echo ================================================
echo    AM/FM Modulation Recognition System
echo ================================================
echo.

echo [1/3] Detecting Python environment...
echo.

set "PYTHON_EXE="

for /f "delims=" %%a in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%a"
    goto :found_python
)

for /f "delims=" %%a in ('where py 2^>nul') do (
    set "PYTHON_EXE=%%a"
    goto :found_python
)

for /f "delims=" %%a in ('dir /b "C:\Users\%username%\AppData\Local\Microsoft\WindowsApps\Python*.exe" 2^>nul') do (
    set "PYTHON_EXE=C:\Users\%username%\AppData\Local\Microsoft\WindowsApps\%%a"
    goto :found_python
)

for /f "delims=" %%a in ('dir /b "C:\Program Files\Python3*\python.exe" 2^>nul') do (
    set "PYTHON_EXE=C:\Program Files\Python3*\%%a"
    goto :found_python
)

for /f "delims=" %%a in ('dir /b "C:\Program Files (x86)\Python3*\python.exe" 2^>nul') do (
    set "PYTHON_EXE=C:\Program Files (x86)\Python3*\%%a"
    goto :found_python
)

:found_python
if not defined PYTHON_EXE (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.8+ first:
    echo 1. Open Microsoft Store
    echo 2. Search for "Python"
    echo 3. Install Python 3.10 or later
    echo.
    echo Or download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Found Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

echo [2/3] Installing/Updating dependencies...
echo.

"%PYTHON_EXE%" -m pip install --upgrade pip -q
"%PYTHON_EXE%" -m pip install numpy matplotlib scipy scikit-learn flask -q

if %errorlevel% equ 0 (
    echo Dependencies installed successfully
) else (
    echo WARNING: Some dependencies may have failed. Retrying with verbose output...
    "%PYTHON_EXE%" -m pip install numpy matplotlib scipy scikit-learn flask
)
echo.

echo [3/3] Starting Flask server...
echo.
echo Server starting on http://%FLASK_HOST%:%FLASK_PORT%
echo.

start "" http://%FLASK_HOST%:%FLASK_PORT%

"%PYTHON_EXE%" %FLASK_APP%

echo.
echo ================================================
echo    Server stopped
echo ================================================
echo.
echo Press any key to exit...
pause >nul