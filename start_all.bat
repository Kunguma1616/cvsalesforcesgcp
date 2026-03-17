@echo off
REM Master Startup Script - Starts Both Backend and Frontend

echo.
echo ============================================================
echo  CV Analysis System - Complete Startup
echo ============================================================
echo.

REM Get the root directory
set ROOT_DIR=%~dp0

echo Starting Backend Service...
echo.

REM Start Backend in a new window
start "Backend - FastAPI" cmd /k "cd /d "%ROOT_DIR%backend" && uvicorn main:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak

echo.
echo Starting Frontend Service...
echo.

REM Start Frontend in a new window
start "Frontend - React" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev"

timeout /t 2 /nobreak

echo.
echo ============================================================
echo ✓ Services Started!
echo ============================================================
echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:3000
echo.
echo Two browser windows will open automatically.
echo Press any key to close this window...
echo ============================================================
echo.

pause

REM Open in browser
start http://localhost:3000

echo.
echo To stop services:
echo   1. Close the Backend window
echo   2. Close the Frontend window
echo.
