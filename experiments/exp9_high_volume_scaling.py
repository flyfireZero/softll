"""
Experiment 9: High-Volume Autoregressive Scaling (6.14M Tokens, 1,500 Steps) on TinyStories
Hypothesis:
Scaling the Soft L-Topological language model to high token volumes (>6M tokens) with seq_len=128
and cosine learning rate decay will:
1. Drive perplexity down into fluent, coherent narrative regimes while sustaining high exact-zero sparsity (>50%).
2. Produce coherent multi-sentence story generations across diverse narrative prompts.
3. Match or surpass standard dot-product Transformer validation loss while eliminating over half
   of all token-to-token connections as mathematical exact zeros (0.0000).
4. Maintain fast KV-cached inference throughput with zero memory leakage.
"""

import os
import json
import math
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM
from soft_ltopology_lm.trainer import ModelTrainer, TrainingConfig
from soft_ltopology_lm.dataset import get_tinystories_train_val_dataloaders


def compute_ngram_diversity(text: str) -> Dict[str, float]:
    r"""
    Computes Distinct-1, Distinct-2, and Distinct-3 diversity metrics.
    Higher values indicate greater lexical diversity and lack of degenerative loops.
    """
    words = [w.lower().strip(".,!?;:\"'()[]") for w in text.split() if w.strip(".,!?;:\"'()[]")]
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0}
        
    unigrams = words
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)]
    
    d1 = len(set(unigrams)) / max(len(unigrams), 1)
    d2 = len(set(bigrams)) / max(len(bigrams), 1) if bigrams else 0.0
    d3 = len(set(trigrams)) / max(len(trigrams), 1) if trigrams else 0.0
    
    return {
        "distinct_1": d1 * 100.0,
        "distinct_2": d2 * 100.0,
        "distinct_3": d3 * 100.0,
    }


def run_experiment_9(max_steps: int = 1500):
    print("=" * 80)
    print(f"   EXPERIMENT 9: HIGH-VOLUME AUTOREGRESSIVE SCALING ({max_steps} STEPS, ~6.14M TOKENS)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Mixed Precision (AMP): Enabled")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    vocab_size = tokenizer.vocab_size
    
    config = TrainingConfig(
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
        seq_len=128,          # Increased context length
        batch_size=8,
        grad_accum_steps=4,   # Effective batch size = 32 sequences = 4,096 tokens/step
        learning_rate=1.5e-3,
        min_learning_rate=1.5e-4,
        weight_decay=0.01,
        max_steps=max_steps,  # 1500 steps = 6,144,000 tokens
        warmup_steps=100,
        use_amp=True,
        log_interval=50,
        eval_interval=250,    # Eval every 250 steps
        checkpoint_interval=500, # Checkpoint every 500 steps
        checkpoint_dir="checkpoints/exp9",
    )
    
    print("\nLoading TinyStories Train & Held-out Validation splits...")
    train_loader, val_loader = get_tinystories_train_val_dataloaders(
        tokenizer,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        val_ratio=0.05,
    )
    print(f"• Train sequences: {len(train_loader.dataset):,} ({len(train_loader.dataset)*config.seq_len:,} tokens)")
    print(f"• Val sequences:   {len(val_loader.dataset):,} ({len(val_loader.dataset)*config.seq_len:,} tokens)")
    
    # 1. Instantiate Models with 1:1 parameter parity
    torch.manual_seed(42)
    soft_l_model = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.seq_len,
        router_type="lukasiewicz",
        bounded_values=False,
    ).to(device)
    
    torch.manual_seed(42)
    baseline_model = StandardTransformerLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.seq_len,
    ).to(device)
    
    print(f"\nModel Parameter Parity:")
    print(f"• Soft L-Topological LM: {soft_l_model.num_parameters:,} parameters")
    print(f"• Baseline Transformer:  {baseline_model.num_parameters:,} parameters")
    
    trainer = ModelTrainer(config=config, device=device)
    
    # Train Soft L-Topology
    t0_soft = time.time()
    res_soft_l = trainer.train(
        soft_l_model,
        train_loader,
        val_dataloader=val_loader,
        desc="[1/2] Soft L-Topology Scaling (1500 Steps)",
    )
    t_soft_total = time.time() - t0_soft
    
    # Train Standard Baseline
    t0_base = time.time()
    res_baseline = trainer.train(
        baseline_model,
        train_loader,
        val_dataloader=val_loader,
        desc="[2/2] Standard Baseline Scaling (1500 Steps)",
    )
    t_base_total = time.time() - t0_base
    
    # Benchmark Multi-Prompt Coherent Generation
    test_prompts = [
        "Once upon a time, there was a little girl named Lily who loved to play with",
        "One sunny day, a little dog named Spot found a big shiny bone in the",
        "Tim and his mother went to the grocery store. Tim wanted to buy a big",
        "The wise little bird flew high into the sky and saw a magical",
        "In a small house near the woods, a kind teddy bear woke up and said",
    ]
    
    print("\n" + "=" * 80)
    print("   COHERENT TEXT GENERATION BENCHMARK (5 DIVERSE PROMPTS)")
    print("=" * 80)
    
    soft_l_model.eval()
    baseline_model.eval()
    
    soft_l_stories = []
    baseline_stories = []
    soft_l_speeds = []
    baseline_speeds = []
    
    for p_idx, prompt in enumerate(test_prompts, 1):
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Soft L-Topology Generation (KV Cached)
        t_start = time.time()
        soft_out = soft_l_model.generate(
            prompt_ids,
            max_new_tokens=65,
            temperature=0.75,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            use_cache=True,
        )
        t_soft_gen = time.time() - t_start
        soft_text = tokenizer.decode(soft_out[0], skip_special_tokens=True)
        soft_speed = 65 / max(t_soft_gen, 1e-4)
        soft_l_stories.append(soft_text)
        soft_l_speeds.append(soft_speed)
        
        # Baseline Generation
        t_start = time.time()
        base_out = baseline_model.generate(
            prompt_ids,
            max_new_tokens=65,
            temperature=0.75,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        t_base_gen = time.time() - t_start
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
        base_speed = 65 / max(t_base_gen, 1e-4)
        baseline_stories.append(base_text)
        baseline_speeds.append(base_speed)
        
        print(f"\n[Prompt {p_idx}/5]: \"{prompt}\"")
        print(f"  • Soft L-Topology ({soft_speed:.1f} tok/s):")
        print(f"    \"{soft_text}\"")
        print(f"  • Baseline ({base_speed:.1f} tok/s):")
        print(f"    \"{base_text}\"")
        
    # Compute n-gram diversity across all generated text
    soft_all_text = " ".join(soft_l_stories)
    base_all_text = " ".join(baseline_stories)
    soft_div = compute_ngram_diversity(soft_all_text)
    base_div = compute_ngram_diversity(base_all_text)
    
    avg_soft_speed = sum(soft_l_speeds) / len(soft_l_speeds)
    avg_base_speed = sum(baseline_speeds) / len(baseline_speeds)
    
    print("\n--- Generation Quality & Diversity Metrics ---")
    print(f"Soft L-Topology: Distinct-1: {soft_div['distinct_1']:.1f}% | Distinct-2: {soft_div['distinct_2']:.1f}% | Distinct-3: {soft_div['distinct_3']:.1f}% | Avg Speed: {avg_soft_speed:.1f} tok/s")
    print(f"Baseline:        Distinct-1: {base_div['distinct_1']:.1f}% | Distinct-2: {base_div['distinct_2']:.1f}% | Distinct-3: {base_div['distinct_3']:.1f}% | Avg Speed: {avg_base_speed:.1f} tok/s")
    
    # 4. Generate Visual Analytics
    os.makedirs("artifacts", exist_ok=True)
    plot_path = "artifacts/exp9_scaling_convergence.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # Panel 1: Training Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(res_soft_l["losses"], label="Soft L-Topology (Łukasiewicz)", color="#1f77b4", alpha=0.85, linewidth=1.5)
    ax1.plot(res_baseline["losses"], label="Standard Transformer Baseline", color="#d62728", alpha=0.85, linewidth=1.5)
    ax1.set_title("Training Loss Convergence (1,500 Steps / 6.14M Tokens)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimization Step", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)
    
    # Panel 2: Validation Loss & Perplexity
    ax2 = axes[0, 1]
    if res_soft_l["val_steps"]:
        ax2.plot(res_soft_l["val_steps"], res_soft_l["val_losses"], label="Soft L-Topology Val Loss", color="#1f77b4", marker="o", linewidth=2.0)
        ax2.plot(res_baseline["val_steps"], res_baseline["val_losses"], label="Baseline Val Loss", color="#d62728", marker="s", linewidth=2.0)
        ax2.set_title("Held-Out Validation Loss Convergence", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Optimization Step", fontsize=10)
        ax2.set_ylabel("Validation Loss", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", frameon=True)
        
    # Panel 3: Deterministic Sparsity Evolution
    ax3 = axes[1, 0]
    ax3.plot(res_soft_l["sparsities"], label="Soft L-Topology (% Exact Zeros)", color="#2ca02c", linewidth=1.5)
    ax3.axhline(0.0, color="#d62728", linestyle="--", label="Standard Transformer (0.00% Exact Zeros)")
    ax3.set_title("Deterministic Sparsity Trajectory (% Exactly Zero Connections)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Optimization Step", fontsize=10)
    ax3.set_ylabel("Exact Zero Connections (%)", fontsize=10)
    ax3.set_ylim(-2.0, 100.0)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="center right", frameon=True)
    
    # Panel 4: Diversity & Inference Speed Comparison
    ax4 = axes[1, 1]
    metrics_names = ["Distinct-1 (%)", "Distinct-2 (%)", "Distinct-3 (%)", "Speed (tok/s / 5)"]
    soft_vals = [soft_div["distinct_1"], soft_div["distinct_2"], soft_div["distinct_3"], avg_soft_speed / 5.0]
    base_vals = [base_div["distinct_1"], base_div["distinct_2"], base_div["distinct_3"], avg_base_speed / 5.0]
    x_pos = range(len(metrics_names))
    width = 0.35
    
    ax4.bar([p - width/2 for p in x_pos], soft_vals, width, label="Soft L-Topology", color="#1f77b4", alpha=0.85)
    ax4.bar([p + width/2 for p in x_pos], base_vals, width, label="Baseline", color="#d62728", alpha=0.85)
    ax4.set_xticks(list(x_pos))
    ax4.set_xticklabels(metrics_names, fontsize=9)
    ax4.set_title("Text Diversity & Generation Speed Profile", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Score", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved scaling benchmark plot to: {plot_path}")
    
    # Copy to brain artifact directory if available
    brain_dir = os.path.expanduser("/home/zero/.gemini/antigravity-cli/brain/ebcbf1a3-eaae-447c-a440-83473140be5a")
    if os.path.exists(brain_dir):
        import shutil
        brain_plot = os.path.join(brain_dir, "exp9_scaling_convergence.png")
        shutil.copyfile(plot_path, brain_plot)
        print(f"Copied visual artifact to: {brain_plot}")
        
    results = {
        "experiment": "Experiment 9: High-Volume Autoregressive Scaling on TinyStories",
        "steps": int(max_steps),
        "effective_batch_size": int(config.batch_size * config.grad_accum_steps),
        "seq_len": int(config.seq_len),
        "total_tokens_trained": int(max_steps * config.batch_size * config.grad_accum_steps * config.seq_len),
        "soft_l": {
            "final_train_loss": float(res_soft_l["final_loss"]),
            "final_train_ppl": float(res_soft_l["final_perplexity"]),
            "best_val_loss": float(res_soft_l["best_val_loss"]),
            "best_val_ppl": float(math.exp(min(res_soft_l["best_val_loss"], 20.0))),
            "final_sparsity_pct": float(res_soft_l["final_sparsity"]),
            "training_tokens_per_sec": float(res_soft_l["tokens_per_second"]),
            "generation_tokens_per_sec": float(avg_soft_speed),
            "diversity": {k: float(v) for k, v in soft_div.items()},
            "sample_stories": soft_l_stories,
        },
        "baseline": {
            "final_train_loss": float(res_baseline["final_loss"]),
            "final_train_ppl": float(res_baseline["final_perplexity"]),
            "best_val_loss": float(res_baseline["best_val_loss"]),
            "best_val_ppl": float(math.exp(min(res_baseline["best_val_loss"], 20.0))),
            "final_sparsity_pct": 0.0,
            "training_tokens_per_sec": float(res_baseline["tokens_per_second"]),
            "generation_tokens_per_sec": float(avg_base_speed),
            "diversity": {k: float(v) for k, v in base_div.items()},
            "sample_stories": baseline_stories,
        },
        "plot_path": plot_path,
    }
    
    with open("experiments/results_exp9.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 80)
    print("   EXPERIMENT 9 COMPLETED SUCCESSFULLY")
    print(f"   Soft L-Topology Val PPL: {results['soft_l']['best_val_ppl']:.2f} | Sparsity: {results['soft_l']['final_sparsity_pct']:.2f}%")
    print(f"   Baseline Val PPL:        {results['baseline']['best_val_ppl']:.2f} | Sparsity: 0.00%")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_experiment_9()
