# Soft L-Topological Language Model: Execution & Evaluation Instructions

This package contains the complete, self-contained codebase, trained model checkpoints (weights), empirical evaluation scripts, unit tests, and pre-tokenized dataset caches for the **Soft L-Topological Language Model**.

---

## 🚀 Quick Setup (Choose Option A or Option B)

### Option A: Using `uv` (Recommended - Fast & Reproducible)
[`uv`](https://github.com/astral-sh/uv) automatically manages Python, virtual environments, and CUDA PyTorch dependencies.

```bash
# 1. Ensure uv is installed
# If not installed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2. Synchronize environment and install locked dependencies
uv sync
```

### Option B: Using Standard Python & `pip`
Requires Python 3.11+.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 2. Install PyTorch with CUDA (or CPU) and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets matplotlib pandas pytest
```

---

## ⚡ 1. Interactive Text Generation (Using Trained Weights)

Generate coherent text step-by-step using the pre-trained 13.67M parameter Soft L-Topological checkpoint (`checkpoints/exp9/1_2__soft_l_topology_scaling__1500_steps_best.pt`):

```bash
# Default prompt with KV-cached decoding and sparsity logging
uv run python generate.py

# Custom prompt and token length
uv run python generate.py --prompt "Once upon a time, Lily found a little bird" --tokens 65 --temp 0.75

# Compare against the Standard Transformer Baseline
uv run python generate.py --model baseline --prompt "Once upon a time, Lily found a little bird"
```

---

## 🧪 2. Run the Verification Test Suite

Run the 19 automated unit tests verifying mathematical operator properties, continuous lattice $[0, 1]^D$ bounds, KV-cache numerical parity ($5.06 \times 10^{-7}$ machine epsilon), parameter parity, and bidirectional encoders:

```bash
# Run via pytest
uv run pytest tests/

# Or run via bash script
bash run_tests.sh
```

---

## 📊 3. Run Experiments & Benchmarks

Each experiment can be run independently:

```bash
# Experiment 1: Baseline Stabilization & Convergence on WikiText-2
uv run python experiments/exp1_convergence.py

# Experiment 2: Lattice Operator Family Ablation (Łukasiewicz, Frank, Gödel, Product)
uv run python experiments/exp2_operators.py

# Experiment 3: Value Projection Dynamics (Bounded [0,1] vs Unbounded R^D)
uv run python experiments/exp3_value_dynamics.py

# Experiment 4: Nilpotent Threshold Sensitivity (tau Grid)
uv run python experiments/exp4_tau_annealing.py

# Experiment 5: Hallucination Resistance & Factual Containment Stress Test
uv run python experiments/exp5_hallucination.py

# Experiment 6: Big Dataset Scaling on TinyStories (4.76M Tokens)
uv run python experiments/exp6_tinystories_scaling.py

# Experiment 7: Distractor Scaling Needle-in-a-Haystack (1 to 16 Distractors)
uv run python experiments/exp7_distractor_stress_test.py

# Experiment 8: Multi-Step Deductive Entailment Benchmark (Modus Ponens Transitivity)
uv run python experiments/exp8_logical_deduction.py

# Experiment 9: High-Volume Autoregressive Scaling (1,500 Steps / 6.14M Tokens)
uv run python experiments/exp9_high_volume_scaling.py

# Experiment 10: Bidirectional Encoder MLM & Distractor Isolation Benchmark
uv run python experiments/exp10_encoder_edge.py
```

---

## 📁 Included Checkpoints & Weights

All model weights are stored in `checkpoints/`:
- **`checkpoints/exp9/`**:
  - `1_2__soft_l_topology_scaling__1500_steps_best.pt`: Best validation checkpoint from the 6.14M token run (Val Loss: 3.7455, Sparsity: 55.06%). Packaged in half-precision FP16 (27.38 MB) for compact release, automatically loaded by `generate.py`.
  - `2_2__standard_baseline_scaling__1500_steps_best.pt`: Best standard transformer baseline validation checkpoint (27.37 MB FP16).
*(Note: Redundant intermediate step checkpoints such as `step_500.pt` and `step_1000.pt` were omitted to ensure the package remains ultra-compact under 100 MB).*

---

## 📑 Documentation & Research Logs

- **[`README.md`](README.md)**: Main project documentation, mathematical derivation of continuous triangular norms, and capstone review matrix.
- **[`EXPERIMENTS.md`](EXPERIMENTS.md)**: Exhaustive research journal logging quantitative results, loss tables, PPL, exact-zero percentages, and qualitative generation traces for all 10 experiments.
- **[`artifacts/`](artifacts/)**: High-resolution visual plots, convergence graphs, sparsity evolution curves, and XAI heatmaps.
