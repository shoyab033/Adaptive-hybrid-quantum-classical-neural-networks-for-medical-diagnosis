@echo off
TITLE Adaptive HQNN Platform Launcher
color 0b

echo ======================================================================
echo          Adaptive HQNN Medical Diagnostic Web Platform
echo ======================================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH! Please install Python 3.9+ and try again.
    pause
    exit /b
)

:: Check for Node / NPM
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js / NPM is not found in PATH! Please install Node.js (LTS) and try again.
    pause
    exit /b
)

echo [1/4] Checking & Installing Python backend dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt

echo.
echo [2/4] Checking & Installing Frontend dependencies...
cd /d "%~dp0frontend"
call npm install

echo.
echo [3/4] Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
cd /d "%~dp0backend"
start "Adaptive HQNN - Backend API (Port 8000)" cmd /k "python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

echo.
echo [4/4] Starting React Frontend on http://localhost:5173 ...
cd /d "%~dp0frontend"
start "Adaptive HQNN - React Frontend (Port 5173)" cmd /k "npm run dev"

echo.
echo ======================================================================
echo  Platform is starting up!
echo.
echo  Backend API:       http://127.0.0.1:8000
echo  API Documentation: http://127.0.0.1:8000/docs
echo  React Frontend:    http://localhost:5173
echo ======================================================================
echo.
echo Opening browser to http://localhost:5173 ...
timeout /t 3 >nul
start http://localhost:5173
