"""
Experiment 2: Operator Comparison (Lukasiewicz vs Parametric vs Product vs Frank vs Godel)
Ablates different differentiable continuous t-norms and t-conorms:
1. Pure Lukasiewicz: T_L(x, y) = max(0, x + y - 1)
2. Parametric Convex: T_lambda(x, y) = lambda * T_L + (1 - lambda) * T_G
3. Product: T_P(x, y) = x * y
4. Frank Family: T_s^F(x, y)
5. Godel: T_G(x, y) = min(x, y)
Plots the Pareto Frontier between Exact Zero Sparsity (%) and Language Model Perplexity.
"""

import os
import json
import math
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from soft_ltopology_lm.model import SoftLTopologicalLM
from soft_ltopology_lm.trainer import ModelTrainer, TrainingConfig, get_wikitext_dataloader


def run_experiment_2():
    print("=" * 80)
    print("   EXPERIMENT 2: DIFFERENTIABLE LATTICE OPERATOR FAMILIES ABLATION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    vocab_size = tokenizer.vocab_size
    
    config = TrainingConfig(
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
        seq_len=64,
        batch_size=16,
        grad_accum_steps=2,
        learning_rate=2e-3,
        min_learning_rate=3e-4,
        max_steps=250,  # 250 steps per operator for clean comparative ablation
        warmup_steps=30,
        log_interval=50,
    )
    
    dataloader = get_wikitext_dataloader(tokenizer, seq_len=config.seq_len, batch_size=config.batch_size)
    trainer = ModelTrainer(config=config, device=device)
    
    operators = [
        ("lukasiewicz", "Łukasiewicz (Strict Nilpotent)"),
        ("parametric", "Parametric Convex (Łukasiewicz + Gödel)"),
        ("product", "Product t-norm (Algebraic)"),
        ("frank", "Frank Family (Smooth Parametric)"),
        ("godel", "Gödel (Min/Max Lattice)"),
    ]
    
    results = {}
    
    for op_key, op_name in operators:
        torch.manual_seed(42)
        model = SoftLTopologicalLM(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            max_seq_len=config.seq_len,
            router_type=op_key,
            bounded_values=True,
        ).to(device)
        
        res = trainer.train(model, dataloader, desc=f"Evaluating Operator: {op_name}")
        results[op_key] = {
            "name": op_name,
            "final_loss": res["final_loss"],
            "perplexity": res["final_perplexity"],
            "exact_zero_sparsity": res["final_sparsity"],
            "tokens_per_second": res["tokens_per_second"],
        }
        
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    with open("experiments/results_exp2.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot Operator Pareto Frontier
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
    markers = ["o", "s", "^", "D", "P"]
    
    for i, (op_key, _) in enumerate(operators):
        r = results[op_key]
        ax.scatter(r["exact_zero_sparsity"], r["perplexity"], color=colors[i], marker=markers[i], s=130, label=r["name"], edgecolors="black", zorder=5)
        ax.annotate(
            f"{op_key}\n(PPL: {r['perplexity']:.1f}, {r['exact_zero_sparsity']:.1f}%)",
            xy=(r["exact_zero_sparsity"], r["perplexity"]),
            xytext=(6, -6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold"
        )
        
    ax.set_xlabel("Exact Zero Routing Sparsity (%) [Higher = More Deterministic Logic]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Perplexity [Lower = Superior Modeling Capacity]", fontsize=10, fontweight="bold")
    ax.set_title("Experiment 2: Pareto Frontier Across Continuous Lattice Operators", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    pareto_path = "artifacts/exp2_operator_pareto.png"
    plt.savefig(pareto_path, bbox_inches="tight")
    plt.close()
    
    print("\n" + "=" * 80)
    print("   EXPERIMENT 2 SUMMARY RESULTS")
    print("=" * 80)
    for op_key, _ in operators:
        r = results[op_key]
        print(f"• {r['name']:<42} | Loss: {r['final_loss']:.4f} | PPL: {r['perplexity']:6.2f} | Sparsity: {r['exact_zero_sparsity']:5.2f}%")
    print(f"• Pareto Plot Saved: {pareto_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_2()
