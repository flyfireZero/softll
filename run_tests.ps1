# PowerShell Test Runner
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   Running Soft L-Topological LM Tests & Demo" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = ".\.venv\Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment not found. Please run .\setup_venv.ps1 first." -ForegroundColor Red
    Exit 1
}

& $pythonExe train_toy.py
