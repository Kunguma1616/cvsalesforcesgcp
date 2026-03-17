@echo off
REM start-backend.bat - Start the Python backend server

echo.
echo ========================================
echo  Fleet App Backend Startup
echo ========================================
echo.

cd backend

REM Check if .env exists
if not exist .env (
    echo.
    echo WARNING: .env file not found in backend directory
    echo Please create backend\.env with the required environment variables
    echo.
)

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:8000...
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
