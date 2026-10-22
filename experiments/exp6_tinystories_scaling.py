"""
Experiment 6: Big Dataset Scaling & KV-Cached Autoregression on TinyStories (4.76M Tokens)
Hypothesis:
When scaled to a high-coherence big dataset (TinyStories, ~4.76M BPE tokens) with fp16 mixed precision:
1. Soft L-Topology converges smoothly to low perplexity with grammatically coherent story generation.
2. Fast KV caching achieves >200 tokens/sec autoregressive generation on consumer GPUs.
3. Deterministic exact-zero routing sparsity remains robust (>30%), proving that language logic
   does not require dense all-to-all softmax connectivity.
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

from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM
from soft_ltopology_lm.trainer import ModelTrainer, TrainingConfig
from soft_ltopology_lm.dataset import get_tinystories_dataloader


def run_experiment_6():
    print("=" * 80)
    print("   EXPERIMENT 6: BIG DATASET SCALING ON TINYSTORIES (4.76M TOKENS)")
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
        seq_len=64,
        batch_size=16,
        grad_accum_steps=2,
        learning_rate=2e-3,
        min_learning_rate=3e-4,
        max_steps=500,  # 500 steps on TinyStories
        warmup_steps=50,
        use_amp=True,   # Enable fp16 mixed precision
        log_interval=50,
        checkpoint_dir="checkpoints/exp6",
    )
    
    print("Loading TinyStories DataLoader from pre-tokenized 4.76M token cache...")
    dataloader = get_tinystories_dataloader(tokenizer, seq_len=config.seq_len, batch_size=config.batch_size)
    print(f"Dataset ready: {len(dataloader.dataset):,} sequence chunks ({len(dataloader.dataset)*config.seq_len:,} tokens)")
    
    # Instantiate Models with 1:1 parameter parity
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
    
    print(f"• Soft L-Topological LM: {soft_l_model.num_parameters:,} parameters")
    print(f"• Baseline Transformer:  {baseline_model.num_parameters:,} parameters")
    
    trainer = ModelTrainer(config=config, device=device)
    
    # 1. Train Soft L-Topology on TinyStories
    res_soft_l = trainer.train(soft_l_model, dataloader, desc="[1/2] Training Soft L-Topology on TinyStories")
    
    # 2. Train Standard Transformer Baseline on TinyStories
    res_baseline = trainer.train(baseline_model, dataloader, desc="[2/2] Training Standard Baseline on TinyStories")
    
    # 3. Benchmark Fast Autoregressive Generation with KV Caching
    prompt_text = "Once upon a time, Lily found a little bird in the garden. The bird was"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    print("\n--- Benchmarking Fast Autoregressive Generation with KV Caching ---")
    soft_l_model.eval()
    baseline_model.eval()
    
    t0 = time.time()
    gen_tokens_s = soft_l_model.generate(prompt_ids, max_new_tokens=40, temperature=0.7, use_cache=True)
    t_gen_s = time.time() - t0
    tok_s_rate = 40 / max(t_gen_s, 1e-4)
    story_soft_l = tokenizer.decode(gen_tokens_s[0])
    
    t0 = time.time()
    gen_tokens_b = baseline_model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
    t_gen_b = time.time() - t0
    tok_b_rate = 40 / max(t_gen_b, 1e-4)
    story_baseline = tokenizer.decode(gen_tokens_b[0])
    
    print(f"• Soft L-Topology Generation Speed: {tok_s_rate:.1f} tokens/sec ({t_gen_s:.3f}s)")
    print(f"• Baseline Transformer Speed:      {tok_b_rate:.1f} tokens/sec ({t_gen_b:.3f}s)")
    print(f"\n[Generated Story - Soft L-Topology]:\n\"{story_soft_l}\"\n")
    print(f"[Generated Story - Baseline Transformer]:\n\"{story_baseline}\"\n")
    
    # 4. Generate Visual Plots
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    steps = range(1, config.max_steps + 1)
    
    # Loss curves
    ax1.plot(steps, res_soft_l["losses"], label=f"Soft L-Topology (Final Loss: {res_soft_l['final_loss']:.3f}, PPL: {res_soft_l['final_perplexity']:.1f})", color="#2ecc71", linewidth=2)
    ax1.plot(steps, res_baseline["losses"], label=f"Standard Transformer (Final Loss: {res_baseline['final_loss']:.3f}, PPL: {res_baseline['final_perplexity']:.1f})", color="#e74c3c", linewidth=2, linestyle="--")
    ax1.set_xlabel("Optimization Steps (Effective Batch = 32)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.set_title("TinyStories Pretraining Loss (4.76M Tokens)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Sparsity curve
    ax2.plot(steps, res_soft_l["sparsities"], color="#27ae60", linewidth=2, label=f"Soft L-Topology Sparsity (Final: {res_soft_l['final_sparsity']:.1f}%)")
    ax2.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.5, label="Standard Transformer (0.00% Exact Zeros)")
    ax2.set_xlabel("Optimization Steps", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Causal Routing Sparsity (% Exact Zeros)", fontsize=10, fontweight="bold")
    ax2.set_ylim(-2, 100)
    ax2.set_title("Deterministic Sparsity Evolution on Big Dataset", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Experiment 6: Scaling to Big Dataset (TinyStories, 13.7M Params)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = "artifacts/exp6_tinystories_scaling.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    results = {
        "experiment": "Experiment 6: Big Dataset Scaling on TinyStories",
        "dataset_tokens": len(dataloader.dataset) * config.seq_len,
        "steps": config.max_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "soft_l": {
            "final_loss": res_soft_l["final_loss"],
            "perplexity": res_soft_l["final_perplexity"],
            "exact_zero_sparsity": res_soft_l["final_sparsity"],
            "training_tokens_per_sec": res_soft_l["tokens_per_second"],
            "generation_tokens_per_sec": tok_s_rate,
            "generated_story": story_soft_l,
        },
        "baseline": {
            "final_loss": res_baseline["final_loss"],
            "perplexity": res_baseline["final_perplexity"],
            "exact_zero_sparsity": 0.0,
            "training_tokens_per_sec": res_baseline["tokens_per_second"],
            "generation_tokens_per_sec": tok_b_rate,
            "generated_story": story_baseline,
        },
        "plot_path": plot_path,
    }
    
    with open("experiments/results_exp6.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("   EXPERIMENT 6 SUMMARY RESULTS")
    print("=" * 80)
    print(f"• Soft L-Topology Final Loss: {res_soft_l['final_loss']:.4f} | PPL: {res_soft_l['final_perplexity']:.2f} | Exact Zeros: {res_soft_l['final_sparsity']:.2f}%")
    print(f"• Baseline Transf Final Loss: {res_baseline['final_loss']:.4f} | PPL: {res_baseline['final_perplexity']:.2f} | Exact Zeros:  0.00%")
    print(f"• Autoregressive Generation:  {tok_s_rate:.1f} tokens/sec with KV caching")
    print(f"• Plot Saved:                 {plot_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_6()
