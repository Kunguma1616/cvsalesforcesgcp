@echo off
REM start-frontend.bat - Start the React frontend

echo.
echo ========================================
echo  Fleet App Frontend Startup
echo ========================================
echo.

cd frontend

REM Check if node_modules exist
if not exist node_modules (
    echo Installing Node dependencies...
    call npm install
    echo.
)

echo Starting Vite development server on http://localhost:3000...
echo.

call npm run dev

pause
