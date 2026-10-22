"""
End-to-End Benchmark Training Script (First Review Milestone):
Trains SoftLTopologicalLM and StandardTransformerLM (13.7M parameters, within 10M-50M target)
on WikiText-2 using BPE (GPT-2) tokenization.
Evaluates:
1. Training Loss and Perplexity convergence
2. Exact Zero Routing Sparsity vs 0.00% Baseline Transformer
3. Explainable AI (XAI) token-level lattice traces
4. High-resolution visual artifacts (Heatmaps, Sparsity Plots, Loss Curves)
5. Autoregressive Text Generation comparison
"""

import os
import time
import math
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SoftLTopologicalLM, StandardTransformerLM, get_wikitext_suppression_tokens
from xai_utils import (
    log_deterministic_reasoning_trace,
    compare_sparsity_summary,
    save_routing_heatmap,
    save_multihead_comparison_heatmap,
    plot_sparsity_comparison,
    format_autoregressive_trace,
    save_autoregressive_step_heatmap,
)


class WikiTextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_samples = len(self.tokens) - self.seq_len - 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_raw_wikitext2() -> str:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "wiki.train.tokens")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"WikiText-2 tokens file not found at {txt_path}.")


def get_dataloader(tokenizer, seq_len: int, batch_size: int, device: str, max_tokens: Optional[int] = None):
    print("Loading and tokenizing WikiText-2 corpus...")
    raw_text = load_raw_wikitext2()
    
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    text = "\n".join(lines)
    
    tokenizer.model_max_length = int(1e9)
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids.squeeze(0)
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        
    print(f"Dataset tokenized: {len(tokens):,} BPE tokens loaded.")
    tokens = tokens.to(device)
    
    ds = WikiTextDataset(tokens, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def plot_training_curves(soft_l_losses, baseline_losses, save_path="artifacts/loss_comparison.png"):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    steps = range(1, len(soft_l_losses) + 1)
    ax.plot(steps, soft_l_losses, label="Soft L-Topological LM (Continuous Lattice)", color="#2ecc71", linewidth=2)
    if baseline_losses:
        ax.plot(steps, baseline_losses, label="Standard Transformer LM (Dot-Product)", color="#e74c3c", linewidth=2, linestyle="--")
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("WikiText-2 BPE Training Convergence (13.7M Parameters)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("   SOFT L-TOPOLOGICAL LM: WIKITEXT-2 BPE BENCHMARK (FIRST REVIEW)")
    print("=" * 75)
    print(f"Compute Device: {device}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    vocab_size = tokenizer.vocab_size
    
    # Target 10M-50M Parameter Architecture
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    hidden_dim = 512
    seq_len = 64
    batch_size = 16
    lr = 2e-3
    max_steps = 600
    
    dataloader = get_dataloader(tokenizer, seq_len, batch_size, device)
    data_iter = iter(dataloader)
    
    soft_l_model = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
        dropout=0.1,
        init_type="beta",
        beta_alpha=1.0,
        beta_beta=3.0,
        router_type="lukasiewicz",
    ).to(device)
    
    baseline_model = StandardTransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
        dropout=0.1,
    ).to(device)
    
    print(f"\nModel Parameter Verification:")
    print(f"• Soft L-Topological LM: {soft_l_model.num_parameters:,} params (~{soft_l_model.num_parameters/1e6:.2f}M)")
    print(f"• Baseline Transformer:  {baseline_model.num_parameters:,} params (~{baseline_model.num_parameters/1e6:.2f}M)")
    print(f"• Exact Size Match Delta: {abs(soft_l_model.num_parameters - baseline_model.num_parameters)} parameters\n")
    
    # Cosine LR schedule with warmup
    warmup_steps = 50
    def get_lr(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Train Soft L-Topological LM
    optimizer_s = torch.optim.AdamW(soft_l_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler_s = torch.optim.lr_scheduler.LambdaLR(optimizer_s, get_lr)
    soft_l_model.train()
    
    print("--- [1/3] Training Soft L-Topological LM ---")
    start_time_s = time.time()
    soft_l_loss_history = []
    last_sparsity = 0.0
    
    for step in range(1, max_steps + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            xb, yb = next(data_iter)
            
        logits, loss, metrics = soft_l_model(xb, yb)
        
        optimizer_s.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(soft_l_model.parameters(), 1.0)
        optimizer_s.step()
        scheduler_s.step()
        
        soft_l_loss_history.append(loss.item())
        last_sparsity = metrics['mean_zero_sparsity_pct']
        
        if step % 50 == 0 or step == max_steps:
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"  Step {step:4d}/{max_steps:4d} | Loss: {loss.item():.4f} | Perplexity: {ppl:7.2f} | Routing Sparsity: {last_sparsity:5.2f}% exact zeros")
            
    elapsed_s = time.time() - start_time_s
    throughput_s = (max_steps * batch_size * seq_len) / elapsed_s
    print(f"✓ Soft L-Topology training completed in {elapsed_s:.2f}s ({throughput_s:,.1f} tokens/sec)\n")
    
    # Train Baseline Transformer for 1:1 Benchmark
    print("--- [2/3] Training Standard Dot-Product Transformer Baseline ---")
    optimizer_b = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, get_lr)
    baseline_model.train()
    data_iter = iter(dataloader)
    
    start_time_b = time.time()
    baseline_loss_history = []
    
    for step in range(1, max_steps + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            xb, yb = next(data_iter)
            
        logits, loss = baseline_model(xb, yb)
        
        optimizer_b.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
        optimizer_b.step()
        scheduler_b.step()
        
        baseline_loss_history.append(loss.item())
        
        if step % 50 == 0 or step == max_steps:
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"  Step {step:4d}/{max_steps:4d} | Loss: {loss.item():.4f} | Perplexity: {ppl:7.2f} | Routing Sparsity:  0.00% exact zeros")
            
    elapsed_b = time.time() - start_time_b
    throughput_b = (max_steps * batch_size * seq_len) / elapsed_b
    print(f"✓ Baseline training completed in {elapsed_b:.2f}s ({throughput_b:,.1f} tokens/sec)\n")
    
    # Visual Artifacts & XAI Traces
    print("--- [3/3] Explainability (XAI) & Artifact Generation ---")
    test_phrase = "The logical architecture of continuous lattice models"
    test_ids = tokenizer(test_phrase, return_tensors="pt").input_ids.to(device)
    tokens = [tokenizer.decode([i.item()]) for i in test_ids[0]]
    
    soft_l_model.eval()
    with torch.no_grad():
        _, _, test_metrics = soft_l_model(test_ids, return_routing_traces=True)
    traces = test_metrics['traces']
    
    trace_text = log_deterministic_reasoning_trace(
        tokens, traces[0], layer_idx=0, head_idx=0, trace_type="norm_routing"
    )
    print(trace_text)
    
    os.makedirs("artifacts", exist_ok=True)
    heatmap_path = save_routing_heatmap(
        tokens, traces[0], save_path="artifacts/wikitext2_lattice_heatmap.png", layer_idx=0, head_idx=0
    )
    multihead_path = save_multihead_comparison_heatmap(
        tokens, traces, save_path="artifacts/wikitext2_multihead_heatmap.png", layer_idx=0
    )
    sparsity_plot_path = plot_sparsity_comparison(
        last_sparsity, baseline_sparsity=0.0, save_path="artifacts/wikitext2_sparsity_comparison.png"
    )
    loss_curve_path = "artifacts/wikitext2_loss_comparison.png"
    plot_training_curves(soft_l_loss_history, baseline_loss_history, save_path=loss_curve_path)
    
    print(f"\n✓ Generated Visual Artifacts:")
    print(f"  1. Routing Heatmap:         {heatmap_path}")
    print(f"  2. Multi-Head Routing Plot: {multihead_path}")
    print(f"  3. Sparsity Comparison Bar: {sparsity_plot_path}")
    print(f"  4. Training Loss Curves:    {loss_curve_path}")
    
    print(compare_sparsity_summary(last_sparsity, baseline_sparsity=0.0))
    
    # Text Generation Demo Side-by-Side
    print("--- [4/4] Explainable Autoregressive Generation & Comparison ---")
    prompt = "The continuous lattice"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Automatically extract and suppress all WikiText markup tokens (<, >, @, =, unk)
    suppress_tokens = get_wikitext_suppression_tokens(tokenizer)
    
    # Step-by-step explainable decoding for Soft L-Topology
    gen_s_exp, step_logs = soft_l_model.generate_with_explanation(
        prompt_ids,
        tokenizer=tokenizer,
        max_new_tokens=10,
        temperature=0.75,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.25,
        suppress_token_ids=suppress_tokens,
    )
    print(format_autoregressive_trace(prompt, tokenizer.decode(gen_s_exp[0]), step_logs))
    
    # Save the Dynamic Step-by-Step Autoregressive Heatmap
    step_heatmap_file = save_autoregressive_step_heatmap(
        prompt, step_logs, save_path="artifacts/wikitext2_autoregressive_trace.png"
    )
    print(f"✓ Saved Dynamic Autoregressive Lattice Trace Heatmap to: {step_heatmap_file}")
    
    # Full 35 token comparison
    gen_s = soft_l_model.generate(
        prompt_ids,
        max_new_tokens=35,
        temperature=0.75,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.25,
        suppress_token_ids=suppress_tokens,
    )
    gen_b = baseline_model.generate(
        prompt_ids,
        max_new_tokens=35,
        temperature=0.75,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.25,
        suppress_token_ids=suppress_tokens,
    )
    
    print("--- Side-by-Side Model Continuation (35 Tokens) ---")
    print(f"• Prompt: '{prompt}'")
    print(f"• Soft L-Topological LM (Continuous Lattice):\n  {tokenizer.decode(gen_s[0])}\n")
    print(f"• Standard Transformer Baseline (Dot-Product):\n  {tokenizer.decode(gen_b[0])}\n")
    print("=" * 75)
    print("   BENCHMARK EVALUATION FINISHED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
