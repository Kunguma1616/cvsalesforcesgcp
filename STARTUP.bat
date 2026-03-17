@echo off
REM Combined startup script for both backend and frontend

echo.
echo ========================================
echo  Fleet App - Full Stack Startup
echo ========================================
echo.
echo This will start both the backend and frontend servers
echo.
echo IMPORTANT:
echo - Make sure you have Python 3.12+ and Node.js 18+ installed
echo - You need two terminal windows open
echo.
echo Step 1: Run this in Terminal 1 - Backend Server
echo   cd backend
echo   streamlit run app.py --server.port=8501
echo.
echo Step 2: Run this in Terminal 2 - Frontend Server  
echo   cd frontend
echo   npm install
echo   npm run dev
echo.
echo Backend: http://localhost:8501
echo Frontend: http://localhost:3000
echo.
pause
