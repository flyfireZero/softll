@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo    Soft L-Topological LM - Environment Setup Script
echo =========================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check the box "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo [INFO] Found %PYTHON_VER%

:: 2. Create Virtual Environment (.venv)
if exist ".venv" (
    echo [INFO] Existing virtual environment found in .venv/
) else (
    echo [INFO] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
)

:: 3. Upgrade pip and install PyTorch
echo [INFO] Upgrading pip and installing PyTorch...
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo    Setup completed successfully!
echo.
echo    To run the training and testing demo, execute:
echo        run_tests.bat
echo    or:
echo        .\.venv\Scripts\activate
echo        python train_toy.py
echo =========================================================
echo.
pause
