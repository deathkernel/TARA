@echo off
setlocal
cd /d "%~dp0"
echo.
echo  TARA Web Workspace
echo  ------------------
echo.
where python >nul 2>nul
if errorlevel 1 (echo Python was not found. Install Python 3.10+ and retry.& pause & exit /b 1)
where npm >nul 2>nul
if errorlevel 1 (echo npm was not found. Install Node.js LTS and retry.& pause & exit /b 1)
echo Checking backend web dependencies...
python -m pip install fastapi uvicorn
if errorlevel 1 (echo Failed to install FastAPI/Uvicorn.& pause & exit /b 1)
if not exist "frontend\node_modules" (
  echo Installing frontend dependencies. First run may take a few minutes...
  pushd frontend
  call npm install
  if errorlevel 1 (popd & echo npm install failed.& pause & exit /b 1)
  popd
)
echo Starting TARA Python API on 127.0.0.1:8000...
start "TARA Backend API" cmd /k "cd /d ""%~dp0"" && python -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
echo Starting TARA React frontend on 127.0.0.1:5173...
start "TARA Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"
echo.
echo TARA is starting. Keep both terminal windows open.
echo Frontend: http://127.0.0.1:5173
echo API health: http://127.0.0.1:8000/api/health
echo.
