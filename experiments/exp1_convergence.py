"""
Experiment 1: Baseline Stabilization & Convergence Scaling (13.7M Model on WikiText-2)
Hypothesis:
Continuous Lattice Sequence Routing with nilpotent Lukasiewicz operators and gradient accumulation
achieves stable cross-entropy convergence comparable to standard Transformers,
while maintaining > 40% exact-zero routing sparsity (where standard attention is 0.00%).
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

from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM, get_wikitext_suppression_tokens
from soft_ltopology_lm.trainer import ModelTrainer, TrainingConfig, get_wikitext_dataloader
from soft_ltopology_lm.evaluator import ModelEvaluator
from soft_ltopology_lm.xai import save_routing_heatmap, plot_sparsity_comparison


def run_experiment_1():
    print("=" * 80)
    print("   EXPERIMENT 1: BASELINE STABILIZATION & CONVERGENCE SCALING (13.7M PARAMS)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
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
        max_steps=400,
        warmup_steps=40,
        log_interval=50,
    )
    
    print("Loading WikiText-2 DataLoader...")
    dataloader = get_wikitext_dataloader(tokenizer, seq_len=config.seq_len, batch_size=config.batch_size)
    
    # 1. Instantiate Models
    soft_l_model = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.seq_len,
        router_type="lukasiewicz",
        bounded_values=True,
    ).to(device)
    
    baseline_model = StandardTransformerLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.seq_len,
    ).to(device)
    
    print(f"• Soft L-Topological LM: {soft_l_model.num_parameters:,} parameters")
    print(f"• Baseline Transformer:  {baseline_model.num_parameters:,} parameters")
    
    trainer = ModelTrainer(config=config, device=device)
    
    # 2. Train Soft L-Topology
    res_soft_l = trainer.train(soft_l_model, dataloader, desc="[1/2] Training Soft L-Topological LM")
    
    # 3. Train Baseline Transformer
    res_baseline = trainer.train(baseline_model, dataloader, desc="[2/2] Training Standard Transformer Baseline")
    
    # 4. Generate Visual Artifacts
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # Convergence Curves
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    steps = range(1, config.max_steps + 1)
    ax.plot(steps, res_soft_l["losses"], label=f"Soft L-Topology (Final Loss: {res_soft_l['final_loss']:.3f}, PPL: {res_soft_l['final_perplexity']:.1f})", color="#2ecc71", linewidth=2)
    ax.plot(steps, res_baseline["losses"], label=f"Standard Transformer (Final Loss: {res_baseline['final_loss']:.3f}, PPL: {res_baseline['final_perplexity']:.1f})", color="#e74c3c", linewidth=2, linestyle="--")
    ax.set_xlabel("Optimization Steps (Effective Batch Size = 32)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("Experiment 1: WikiText-2 Convergence Trajectory (13.7M Parameters)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_curve_path = "artifacts/exp1_convergence_curves.png"
    plt.savefig(loss_curve_path, bbox_inches="tight")
    plt.close()
    
    # Sparsity Evolution Curve
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, res_soft_l["sparsities"], color="#27ae60", linewidth=2, label="Soft L-Topology Routing Sparsity (% Exact Zeros)")
    ax.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.5, label="Standard Softmax Baseline (0.00% Exact Zeros)")
    ax.set_xlabel("Optimization Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exact Zero Sparsity (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(-2, 100)
    ax.set_title("Experiment 1: Deterministic Sparsity Evolution Across Training", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    sparsity_curve_path = "artifacts/exp1_sparsity_evolution.png"
    plt.savefig(sparsity_curve_path, bbox_inches="tight")
    plt.close()
    
    # Sparsity Bar
    sparsity_bar_path = plot_sparsity_comparison(
        res_soft_l["final_sparsity"], baseline_sparsity=0.0, save_path="artifacts/exp1_sparsity_bar.png"
    )
    
    # 5. Text Generation Demo
    prompt = "The continuous lattice"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    suppress_tokens = get_wikitext_suppression_tokens(tokenizer)
    
    gen_s = soft_l_model.generate(prompt_ids, max_new_tokens=30, suppress_token_ids=suppress_tokens)
    gen_b = baseline_model.generate(prompt_ids, max_new_tokens=30, suppress_token_ids=suppress_tokens)
    
    text_s = tokenizer.decode(gen_s[0])
    text_b = tokenizer.decode(gen_b[0])
    
    results = {
        "experiment": "Experiment 1: Baseline Stabilization & Convergence Scaling",
        "parameters": soft_l_model.num_parameters,
        "steps": config.max_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "soft_l": {
            "final_loss": res_soft_l["final_loss"],
            "final_perplexity": res_soft_l["final_perplexity"],
            "final_sparsity_pct": res_soft_l["final_sparsity"],
            "tokens_per_second": res_soft_l["tokens_per_second"],
            "sample_generation": text_s,
        },
        "baseline": {
            "final_loss": res_baseline["final_loss"],
            "final_perplexity": res_baseline["final_perplexity"],
            "final_sparsity_pct": 0.0,
            "tokens_per_second": res_baseline["tokens_per_second"],
            "sample_generation": text_b,
        },
        "sparsity_gain_pct": res_soft_l["final_sparsity"],
    }
    
    with open("experiments/results_exp1.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("   EXPERIMENT 1 SUMMARY RESULTS")
    print("=" * 80)
    print(f"• Soft L-Topology Final Loss: {res_soft_l['final_loss']:.4f} | PPL: {res_soft_l['final_perplexity']:.2f} | Exact Zeros: {res_soft_l['final_sparsity']:.2f}%")
    print(f"• Baseline Transf Final Loss: {res_baseline['final_loss']:.4f} | PPL: {res_baseline['final_perplexity']:.2f} | Exact Zeros:  0.00%")
    print(f"• Sparsity Advantage:         +{res_soft_l['final_sparsity']:.2f}% true mathematical zeros")
    print(f"• Artifacts Saved:            {loss_curve_path}, {sparsity_curve_path}, {sparsity_bar_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_1()
