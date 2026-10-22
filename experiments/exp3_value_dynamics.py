"""
Experiment 3: Value Projection Dynamics (Bounded V in [0, 1] vs Unbounded V in R^D)
Hypothesis:
While queries and keys (Q, K) MUST reside in [0, 1] to govern lattice logic (Meet and Join),
allowing value vectors V to remain unbounded linear features in R^D eliminates gradient squashing
from sigmoid squashing, boosting representation capacity and lowering perplexity
WITHOUT diminishing exact-zero routing sparsity.
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


def run_experiment_3():
    print("=" * 80)
    print("   EXPERIMENT 3: VALUE PROJECTION DYNAMICS (BOUNDED vs UNBOUNDED V)")
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
        max_steps=300,
        warmup_steps=30,
        log_interval=50,
    )
    
    dataloader = get_wikitext_dataloader(tokenizer, seq_len=config.seq_len, batch_size=config.batch_size)
    trainer = ModelTrainer(config=config, device=device)
    
    # Model 1: Bounded Values (V in [0, 1] via Sigmoid)
    print("\n--- [1/2] Training Model A: Bounded Values V in [0, 1] ---")
    torch.manual_seed(42)
    model_bounded = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.seq_len,
        router_type="lukasiewicz",
        bounded_values=True,
    ).to(device)
    res_bounded = trainer.train(model_bounded, dataloader, desc="Model A: Bounded V in [0, 1]")
    
    # Model 2: Unbounded Values (V in R^D linear)
    print("\n--- [2/2] Training Model B: Unbounded Values V in R^D ---")
    torch.manual_seed(42)
    model_unbounded = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.seq_len,
        router_type="lukasiewicz",
        bounded_values=False,
    ).to(device)
    res_unbounded = trainer.train(model_unbounded, dataloader, desc="Model B: Unbounded V in R^D")
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # Plot Comparison Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    steps = range(1, config.max_steps + 1)
    
    # Loss plot
    ax1.plot(steps, res_bounded["losses"], label=f"Bounded V in [0, 1] (Final Loss: {res_bounded['final_loss']:.3f}, PPL: {res_bounded['final_perplexity']:.1f})", color="#e67e22", linewidth=2)
    ax1.plot(steps, res_unbounded["losses"], label=f"Unbounded V in R^D (Final Loss: {res_unbounded['final_loss']:.3f}, PPL: {res_unbounded['final_perplexity']:.1f})", color="#2ecc71", linewidth=2)
    ax1.set_xlabel("Optimization Steps", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.set_title("Training Loss Convergence", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Sparsity plot
    ax2.plot(steps, res_bounded["sparsities"], label=f"Bounded V (Sparsity: {res_bounded['final_sparsity']:.1f}%)", color="#e67e22", linewidth=2)
    ax2.plot(steps, res_unbounded["sparsities"], label=f"Unbounded V (Sparsity: {res_unbounded['final_sparsity']:.1f}%)", color="#2ecc71", linewidth=2)
    ax2.set_xlabel("Optimization Steps", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Routing Sparsity (% Exact Zeros)", fontsize=10, fontweight="bold")
    ax2.set_title("Exact Zero Sparsity Retention", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Experiment 3: Value Projection Dynamics (Bounded vs Unbounded V)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    comp_path = "artifacts/exp3_value_dynamics.png"
    plt.savefig(comp_path, bbox_inches="tight")
    plt.close()
    
    results = {
        "experiment": "Experiment 3: Value Projection Dynamics",
        "bounded_v": {
            "final_loss": res_bounded["final_loss"],
            "perplexity": res_bounded["final_perplexity"],
            "exact_zero_sparsity": res_bounded["final_sparsity"],
            "tokens_per_second": res_bounded["tokens_per_second"],
        },
        "unbounded_v": {
            "final_loss": res_unbounded["final_loss"],
            "perplexity": res_unbounded["final_perplexity"],
            "exact_zero_sparsity": res_unbounded["final_sparsity"],
            "tokens_per_second": res_unbounded["tokens_per_second"],
        },
        "ppl_improvement": res_bounded["final_perplexity"] - res_unbounded["final_perplexity"],
        "sparsity_impact": res_unbounded["final_sparsity"] - res_bounded["final_sparsity"],
    }
    
    with open("experiments/results_exp3.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("   EXPERIMENT 3 SUMMARY RESULTS")
    print("=" * 80)
    print(f"• Bounded V in [0, 1]:  Loss: {res_bounded['final_loss']:.4f} | PPL: {res_bounded['final_perplexity']:6.2f} | Sparsity: {res_bounded['final_sparsity']:5.2f}%")
    print(f"• Unbounded V in R^D:   Loss: {res_unbounded['final_loss']:.4f} | PPL: {res_unbounded['final_perplexity']:6.2f} | Sparsity: {res_unbounded['final_sparsity']:5.2f}%")
    print(f"• Plot Saved:           {comp_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_3()
