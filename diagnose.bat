@echo off
REM Diagnostic & Troubleshooting Script

cls
echo.
echo ==========================================
echo  System Diagnostic Tool
echo ==========================================
echo.

setlocal enabledelayedexpansion

REM Check Python installation
echo [CHECK 1] Python Installation
python --version
if errorlevel 1 (
    echo ✗ Python not found or not in PATH
    echo.
    echo FIX: Install Python 3.10+ from python.org and add to PATH
    goto end
) else (
    echo ✓ Python installed
)
echo.

REM Check pip
echo [CHECK 2] PIP Package Manager
pip --version
if errorlevel 1 (
    echo ✗ pip not found
    echo.
    echo FIX: Make sure Python was installed with pip
    goto end
) else (
    echo ✓ pip found
)
echo.

REM Check ports
echo [CHECK 3] Port Availability
echo.
echo Backend port (8000):
netstat -ano | findstr :8000
if errorlevel 1 (
    echo ✓ Port 8000 is free
) else (
    echo ✗ Port 8000 is in use!
    echo FIX: Kill process using port 8000 or change port
)
echo.

echo Frontend port (3000):
netstat -ano | findstr :3000
if errorlevel 1 (
    echo ✓ Port 3000 is free
) else (
    echo ✗ Port 3000 is in use!
    echo FIX: Kill process using port 3000 or change port
)
echo.

REM Check if required files exist
echo [CHECK 4] Required Files
if exist "frontend\package.json" (
    echo ✓ Frontend package.json found
) else (
    echo ✗ Frontend package.json missing
)

if exist "backend\main.py" (
    echo ✓ Backend main.py found
) else (
    echo ✗ Backend main.py missing
)

if exist "backend\requirements.txt" (
    echo ✓ Backend requirements.txt found
) else (
    echo ✗ Backend requirements.txt missing
)
echo.

REM Test backend imports
echo [CHECK 5] Backend Imports
cd /d "%~dp0\backend"
python -c "from main import app; print('OK')" 2>nul
if errorlevel 1 (
    echo ✗ Backend import failed
    echo.
    echo Detailed error:
    python -c "from main import app"
    echo.
    echo FIX: Run 'pip install -r requirements.txt --upgrade'
) else (
    echo ✓ Backend imports successfully
)
echo.

REM Check critical dependencies
echo [CHECK 6] Critical Dependencies
cd /d "%~dp0\backend"
python verify_dependencies.py
echo.

REM System info
echo [CHECK 7] System Information
echo OS: Windows
echo Architecture:
systeminfo | findstr /C:"System Type"
echo.

:end
echo ==========================================
echo Diagnostic complete. See results above.
echo ==========================================
echo.
pause
