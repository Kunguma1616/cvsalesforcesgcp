@echo off
REM Backend Startup Script with Dependency Verification

cls
echo.
echo ========================================
echo  CV Analysis Backend Startup (FastAPI)
echo ========================================
echo.

REM Navigate to backend directory
cd /d "%~dp0"

REM Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found!
    echo.
    echo Make sure you run this from the backend/ directory:
    echo   cd backend
    echo   start_backend.bat
    echo.
    pause
    exit /b 1
)

REM Check Python version
echo [1/4] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo ✓ Python found
echo.

REM Check if .env exists
if not exist ".env" (
    echo [2/4] .env file not found
    echo.
    echo ⚠️  WARNING: .env file needs to be created!
    echo Please create backend\.env with:
    echo   GROQ_API_KEY=your_api_key_here
    echo   AZURE_STORAGE_CONNECTION_STRING=...
    echo   AZURE_SAS_TOKEN=...
    echo   SF_USERNAME=...
    echo.
    echo See .env.example for full template
    echo.
) else (
    echo [2/4] .env file found ✓
)
echo.

REM Verify dependencies
echo [3/4] Verifying dependencies...
python verify_dependencies.py
if errorlevel 1 (
    echo.
    echo Installing missing packages (this may take a minute)...
    pip install -r requirements.txt --upgrade --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)
echo ✓ Dependencies ready
echo.

REM Start the backend
echo [4/4] Starting FastAPI backend...
echo.
echo ========================================
echo Server will start at:
echo   HTTP:  http://127.0.0.1:8000
echo   DOCS:  http://127.0.0.1:8000/docs
echo ========================================
echo.
echo Press CTRL+C to stop the server
echo.

timeout /t 2 /nobreak
uvicorn main:app --host 127.0.0.1 --port 8000

echo.
echo Backend stopped.
pause
