@echo off
REM Start Frontend Server

echo.
echo ========================================
echo  Starting CV Analysis Frontend (React)
echo ========================================
echo.

REM Navigate to frontend directory
cd /d "%~dp0"

REM Install dependencies if not already installed
if not exist "node_modules" (
    echo Installing npm dependencies...
    call npm install
)

echo.
echo ✓ Starting Frontend on http://localhost:3000
echo.
echo Press CTRL+C to stop
echo.

REM Start the development server
call npm run dev

pause
