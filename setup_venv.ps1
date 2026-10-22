# PowerShell Setup Script for Soft L-Topological LM
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   Soft L-Topological LM - PowerShell Setup Script" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Python availability
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "[INFO] Detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/downloads/ and ensure 'Add Python to PATH' is checked." -ForegroundColor Yellow
    Exit 1
}

# 2. Setup Virtual Environment
if (Test-Path -Path ".venv") {
    Write-Host "[INFO] Existing virtual environment found in .venv/" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
    Write-Host "[INFO] Virtual environment created." -ForegroundColor Green
}

# 3. Activate and Install Dependencies
Write-Host "[INFO] Activating virtual environment and installing PyTorch..." -ForegroundColor Cyan
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install torch

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================================" -ForegroundColor Green
    Write-Host "   Setup Completed Successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "   To run tests & demo:" -ForegroundColor White
    Write-Host "       .\run_tests.ps1" -ForegroundColor Yellow
    Write-Host "   or:" -ForegroundColor White
    Write-Host "       .\.venv\Scripts\Activate.ps1; python train_toy.py" -ForegroundColor Yellow
    Write-Host "=========================================================" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Package installation failed." -ForegroundColor Red
}
