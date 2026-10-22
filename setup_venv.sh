#!/usr/bin/env bash
set -e

echo "========================================================="
echo "   Soft L-Topological LM - Linux/WSL Setup Script"
echo "========================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[INFO] Activating virtual environment and installing PyTorch..."
source .venv/bin/activate
pip install --upgrade pip
pip install torch

echo ""
echo "========================================================="
echo "   Setup Completed Successfully!"
echo "   To run tests & demo: python train_toy.py"
echo "========================================================="
