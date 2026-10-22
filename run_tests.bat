@echo off
echo =========================================================
echo    Running Soft L-Topological LM Tests and Demo
echo =========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup_venv.bat first!
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe train_toy.py

echo.
pause
