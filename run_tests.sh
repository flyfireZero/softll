#!/usr/bin/env bash
set -e

echo "========================================================="
echo "   Soft L-Topological LM - First Review Test Suite"
echo "========================================================="
echo ""

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found (.venv/bin/python)."
    exit 1
fi

echo "--- [1/2] Running Character-Level Logic Unit Test (Phase 1) ---"
.venv/bin/python train_toy.py
echo ""

echo "--- [2/2] Running 13.7M Parameter BPE WikiText-2 Benchmark (Phase 2) ---"
.venv/bin/python test.py
echo ""

echo "========================================================="
echo "   ALL TESTS AND BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "========================================================="
