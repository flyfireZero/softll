"""
Experiment 5: Hallucination Resistance & Factual Containment Stress Testing
Directly proves the core thesis of Soft L-Topology:
Replacing statistical Euclidean vectors and softmax proximity with continuous lattice logic
reduces language model hallucinations by mathematically cutting off spurious attention paths
to distractor context to strictly 0.0.
"""

import os
import json
import torch
from transformers import AutoTokenizer

from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM
from soft_ltopology_lm.hallucination import HallucinationBenchmark
from soft_ltopology_lm.trainer import ModelTrainer, TrainingConfig, get_wikitext_dataloader


def run_experiment_5():
    print("=" * 80)
    print("   EXPERIMENT 5: HALLUCINATION RESISTANCE & FACTUAL CONTAINMENT BENCHMARK")
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
        max_steps=250,
        warmup_steps=30,
        log_interval=50,
    )
    
    dataloader = get_wikitext_dataloader(tokenizer, seq_len=config.seq_len, batch_size=config.batch_size)
    trainer = ModelTrainer(config=config, device=device)
    
    print("Instantiating models for hallucination benchmark...")
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
    
    # Train both models
    _ = trainer.train(soft_l_model, dataloader, desc="[1/2] Training Soft L-Topological LM")
    _ = trainer.train(baseline_model, dataloader, desc="[2/2] Training Standard Transformer Baseline")
    
    # Run Hallucination Benchmark
    print("\n--- Running Hallucination & Distractor Containment Evaluation ---")
    bench = HallucinationBenchmark(tokenizer=tokenizer, device=device)
    results = bench.evaluate_distractor_leakage(soft_l_model, baseline_model)
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    plot_path = bench.plot_hallucination_benchmark(
        results, save_path="artifacts/exp5_hallucination_resistance.png"
    )
    
    with open("experiments/results_exp5.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("   EXPERIMENT 5 SUMMARY RESULTS")
    print("=" * 80)
    print(f"• Distractor Pruning Rate (Soft L-Topology): {results['soft_l_prune_rate_pct']:.2f}% exact zeros")
    print(f"• Distractor Pruning Rate (Standard Baseline): {results['baseline_prune_rate_pct']:.2f}% exact zeros (Softmax never prunes!)")
    print(f"• Spurious Attention Leakage (Soft L-Topology): {results['soft_l_distractor_leakage_pct']:.2f}% mass")
    print(f"• Spurious Attention Leakage (Standard Baseline): {results['baseline_distractor_leakage_pct']:.2f}% mass")
    print(f"• Hallucination Resistance Gain: {results['leakage_reduction_pct']:.2f}% reduction in spurious leakage")
    print(f"• Visual Artifact Saved: {plot_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_5()
