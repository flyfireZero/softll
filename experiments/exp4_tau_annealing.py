"""
Experiment 4: Nilpotent Threshold Sensitivity Analysis (tau)
Systematically evaluates the effect of the nilpotent threshold tau on the balance between:
1. Deterministic exact-zero routing sparsity (%)
2. Perplexity (predictive coherence)
Tests tau in [0.0005, 0.0012, 0.0025, 0.0050, 0.0100]
"""

import os
import json
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from soft_ltopology_lm.model import SoftLTopologicalLM
from soft_ltopology_lm.trainer import ModelTrainer, TrainingConfig, get_wikitext_dataloader


def run_experiment_4():
    print("=" * 80)
    print("   EXPERIMENT 4: NILPOTENT THRESHOLD SENSITIVITY (tau ABLATION)")
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
        max_steps=200,
        warmup_steps=25,
        log_interval=50,
    )
    
    dataloader = get_wikitext_dataloader(tokenizer, seq_len=config.seq_len, batch_size=config.batch_size)
    trainer = ModelTrainer(config=config, device=device)
    
    tau_values = [0.0005, 0.0012, 0.0025, 0.0050, 0.0100]
    results = {}
    
    for tau in tau_values:
        torch.manual_seed(42)
        model = SoftLTopologicalLM(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            max_seq_len=config.seq_len,
            router_type="lukasiewicz",
            bounded_values=False,
            init_tau=tau,
        ).to(device)
        
        desc = f"Evaluating Nilpotent Threshold tau = {tau:.4f}"
        res = trainer.train(model, dataloader, desc=desc)
        
        results[str(tau)] = {
            "tau": tau,
            "final_loss": res["final_loss"],
            "perplexity": res["final_perplexity"],
            "exact_zero_sparsity": res["final_sparsity"],
        }
        
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    with open("experiments/results_exp4.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot Sensitivity Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    taus = [results[k]["tau"] for k in results]
    sparsities = [results[k]["exact_zero_sparsity"] for k in results]
    perplexities = [results[k]["perplexity"] for k in results]
    
    ax1.plot(taus, sparsities, marker="o", color="#2ecc71", linewidth=2.5, markersize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Nilpotent Threshold tau (log scale)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Exact Zero Sparsity (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Routing Sparsity vs Threshold tau", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    for t, s in zip(taus, sparsities):
        ax1.annotate(f"{s:.1f}%", xy=(t, s), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")
        
    ax2.plot(taus, perplexities, marker="s", color="#e74c3c", linewidth=2.5, markersize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Nilpotent Threshold tau (log scale)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Perplexity", fontsize=10, fontweight="bold")
    ax2.set_title("Perplexity vs Threshold tau", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    for t, p in zip(taus, perplexities):
        ax2.annotate(f"{p:.1f}", xy=(t, p), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")
        
    plt.suptitle("Experiment 4: Nilpotent Threshold Sensitivity Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    tau_plot_path = "artifacts/exp4_tau_sensitivity.png"
    plt.savefig(tau_plot_path, bbox_inches="tight")
    plt.close()
    
    print("\n" + "=" * 80)
    print("   EXPERIMENT 4 SUMMARY RESULTS")
    print("=" * 80)
    for k in results:
        r = results[k]
        print(f"• tau = {r['tau']:.4f} | Loss: {r['final_loss']:.4f} | PPL: {r['perplexity']:6.2f} | Sparsity: {r['exact_zero_sparsity']:5.2f}%")
    print(f"• Sensitivity Plot Saved: {tau_plot_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_4()
