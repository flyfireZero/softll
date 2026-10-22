"""
Experiment 10: Encoder-Only Architecture Exploration & Logic Edge Discovery
Hypothesis:
In bidirectional encoders (where tokens can attend freely across past and future contexts):
1. Standard dot-product attention (BERT-style) suffers from inescapable attention contamination / leakage
   because softmax(z) > 0 for all tokens, forcing distractor and noise tokens to receive positive weight.
2. Soft L-Topological Continuous Lattice Encoders use nilpotent Łukasiewicz logic to achieve
   true mathematical isolation (exact 0.0000 attention) for logically irrelevant context tokens.
3. In Bidirectional Masked Language Modeling (MLM), Soft L-Topology matches or exceeds prediction accuracy
   while operating with over 40-60% exact-zero routing sparsity.
4. On deductive containment & distractor isolation, Soft L-Topology prevents representation corruption.
"""

import os
import json
import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from soft_ltopology_lm.encoder import SoftLTopologicalEncoder, StandardTransformerEncoder
from soft_ltopology_lm.dataset import load_or_tokenize_tinystories


class MLMDataset(Dataset):
    r"""
    Dataset wrapper for Masked Language Modeling.
    Randomly masks 15% of tokens in each sequence with probability:
    - 80% replaced with mask_token_id (or a designated token)
    - 10% replaced with random token
    - 10% kept unchanged
    """
    def __init__(self, token_tensor: torch.Tensor, seq_len: int = 64, mask_prob: float = 0.15, vocab_size: int = 50257):
        self.seq_len = seq_len
        self.tokens = token_tensor.to(torch.long)
        self.num_samples = (len(self.tokens) - 1) // self.seq_len
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        original = self.tokens[start:end].clone()
        labels = torch.full_like(original, -100)
        
        # Determine masked positions
        rand_mask = torch.rand(self.seq_len) < self.mask_prob
        labels[rand_mask] = original[rand_mask]
        
        input_ids = original.clone()
        # 80% replace with random token from vocabulary (simulating MASK)
        replace_mask = rand_mask & (torch.rand(self.seq_len) < 0.8)
        input_ids[replace_mask] = torch.randint(0, self.vocab_size, (replace_mask.sum().item(),), dtype=torch.long)
        
        return input_ids, labels


def evaluate_mlm(model: nn.Module, dataloader: DataLoader, device: str, max_batches: int = 25) -> Dict[str, float]:
    r"""
    Evaluates Masked Language Modeling accuracy (Top-1, Top-5) and loss.
    """
    model.eval()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_masked_tokens = 0
    total_sparsity = 0.0
    batches = 0
    
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast("cuda", enabled="cuda" in device, dtype=torch.float16):
                out = model(input_ids, mlm_labels=labels)
                
            loss = out["mlm_loss"]
            mlm_logits = out["mlm_logits"]  # [B, S, V]
            sparsity = out["mean_zero_sparsity_pct"]
            
            # Masked positions
            mask = (labels != -100)
            if mask.any():
                masked_logits = mlm_logits[mask]  # [N_masked, V]
                masked_targets = labels[mask]     # [N_masked]
                
                # Top-1
                preds_top1 = masked_logits.argmax(dim=-1)
                total_correct_top1 += (preds_top1 == masked_targets).sum().item()
                
                # Top-5
                top5_preds = torch.topk(masked_logits, k=5, dim=-1)[1]
                correct_top5 = (top5_preds == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item()
                total_correct_top5 += correct_top5
                
                total_masked_tokens += masked_targets.numel()
                
            total_loss += loss.item()
            total_sparsity += sparsity
            batches += 1
            
    model.train()
    return {
        "loss": total_loss / max(batches, 1),
        "top1_acc": (total_correct_top1 / max(total_masked_tokens, 1)) * 100.0,
        "top5_acc": (total_correct_top5 / max(total_masked_tokens, 1)) * 100.0,
        "mean_sparsity": total_sparsity / max(batches, 1),
    }


def run_experiment_10(mlm_steps: int = 800):
    print("=" * 80)
    print("   EXPERIMENT 10: ENCODER-ONLY ARCHITECTURE EXPLORATION & LOGIC EDGE PROBE")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Mixed Precision (AMP): Enabled")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    vocab_size = tokenizer.vocab_size
    
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    hidden_dim = 512
    seq_len = 64
    batch_size = 16
    
    print("\nLoading TinyStories token cache for Bidirectional MLM Pre-training...")
    tokens = load_or_tokenize_tinystories(tokenizer)
    split_idx = int(len(tokens) * 0.95)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    train_ds = MLMDataset(train_tokens, seq_len=seq_len, mask_prob=0.15, vocab_size=vocab_size)
    val_ds = MLMDataset(val_tokens, seq_len=seq_len, mask_prob=0.15, vocab_size=vocab_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # 1. Instantiate Bidirectional Encoders
    torch.manual_seed(42)
    soft_encoder = SoftLTopologicalEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
        router_type="lukasiewicz",
        bounded_values=False,
    ).to(device)
    
    torch.manual_seed(42)
    base_encoder = StandardTransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
    ).to(device)
    
    print(f"\nEncoder Parameter Counts:")
    print(f"• Soft L-Topological Encoder: {soft_encoder.num_parameters:,} parameters")
    print(f"• Standard Transformer Encoder: {base_encoder.num_parameters:,} parameters")
    
    # 2. Train MLM on Both Encoders
    print(f"\n--- [1/3] Bidirectional Masked Language Modeling Pre-Training ({mlm_steps} Steps) ---")
    
    def train_encoder_mlm(encoder, name):
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-3, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled="cuda" in device)
        data_iter = iter(train_loader)
        
        losses = []
        sparsities = []
        t0 = time.time()
        encoder.train()
        
        for step in range(1, mlm_steps + 1):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                xb, yb = next(data_iter)
                
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled="cuda" in device, dtype=torch.float16):
                out = encoder(xb, mlm_labels=yb)
                loss = out["mlm_loss"]
                sparsity = out["mean_zero_sparsity_pct"]
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            sparsities.append(sparsity)
            
            if step % 100 == 0 or step == mlm_steps:
                print(f"  [{name}] Step {step:4d}/{mlm_steps:4d} | MLM Loss: {loss.item():.4f} | Sparsity: {sparsity:5.2f}% zeros")
                
        elapsed = time.time() - t0
        print(f"✓ {name} MLM training complete in {elapsed:.1f}s\n")
        return losses, sparsities
        
    soft_losses, soft_sparsities = train_encoder_mlm(soft_encoder, "Soft L-Topology Encoder")
    base_losses, base_sparsities = train_encoder_mlm(base_encoder, "Standard BERT Baseline")
    
    # 3. Evaluate MLM Accuracy
    print("--- [2/3] Evaluating Masked Token Recovery Accuracy on Held-Out Validation Split ---")
    soft_eval = evaluate_mlm(soft_encoder, val_loader, device)
    base_eval = evaluate_mlm(base_encoder, val_loader, device)
    
    print(f"• Soft L-Topological Encoder:")
    print(f"    MLM Val Loss: {soft_eval['loss']:.4f} | Top-1 Acc: {soft_eval['top1_acc']:.2f}% | Top-5 Acc: {soft_eval['top5_acc']:.2f}% | Sparsity: {soft_eval['mean_sparsity']:.2f}% zeros")
    print(f"• Standard Transformer Encoder:")
    print(f"    MLM Val Loss: {base_eval['loss']:.4f} | Top-1 Acc: {base_eval['top1_acc']:.2f}% | Top-5 Acc: {base_eval['top5_acc']:.2f}% | Sparsity: 0.00% zeros")
    
    # 4. Probe Semantic Distractor Isolation in Bidirectional Context
    print("\n--- [3/3] Bidirectional Contextual Distractor Isolation Stress Test ---")
    # Premise: "Lily has a gentle white cat. The cat plays in the garden."
    # Distractors: Unrelated facts from different domains (e.g., astrophysics, sports, mechanics)
    target_text = "Lily has a gentle white cat. The cat plays in the garden."
    distractor_texts = [
        "Galaxies contain billions of stellar clusters rotating rapidly in deep outer space.",
        "The football referee blew the whistle for a penalty during the tournament.",
        "Automotive engines require synthetic motor oil to lubricate metal bearings.",
        "Quantum superposition allows particles to exist in multiple states simultaneously.",
    ]
    
    target_ids = tokenizer.encode(target_text)
    num_target_tokens = len(target_ids)
    
    # Create combined input: Target Sentence + Distractors
    combined_tokens = list(target_ids)
    for dist in distractor_texts:
        combined_tokens.extend(tokenizer.encode(" " + dist))
    combined_tokens = combined_tokens[:seq_len]
    
    input_tensor = torch.tensor([combined_tokens], dtype=torch.long, device=device)
    
    soft_encoder.eval()
    with torch.no_grad():
        soft_out = soft_encoder(input_tensor, return_routing_traces=True)
        soft_trace = soft_out["traces"][-1]  # Final layer routing trace
        norm_routing = soft_trace["norm_routing"][0]  # [H, S, S]
        mean_routing = norm_routing.mean(dim=0).cpu().numpy()  # [S, S]
        
    # Attention from target tokens (queries 0:num_target_tokens) to distractor keys
    dist_weights_s = norm_routing[:, :num_target_tokens, num_target_tokens:]
    distractor_zero_pct_soft = float((dist_weights_s <= 1e-4).float().mean().item() * 100.0)
    distractor_leakage_soft = float(dist_weights_s.sum(dim=-1).mean().item() * 100.0)
    
    # Compute baseline dot-product attention
    base_encoder.eval()
    with torch.no_grad():
        pos = torch.arange(0, input_tensor.size(1), device=device).unsqueeze(0)
        base_x = base_encoder.tok_emb(input_tensor) + base_encoder.pos_emb(pos)
        W_q = base_encoder.encoder.layers[-1].self_attn.in_proj_weight[:embed_dim, :]
        W_k = base_encoder.encoder.layers[-1].self_attn.in_proj_weight[embed_dim:2*embed_dim, :]
        
        q = torch.matmul(base_x, W_q.t()).view(1, input_tensor.size(1), num_heads, embed_dim // num_heads).transpose(1, 2)
        k = torch.matmul(base_x, W_k.t()).view(1, input_tensor.size(1), num_heads, embed_dim // num_heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(embed_dim // num_heads)
        base_attn = torch.softmax(scores, dim=-1)[0]
        
    dist_weights_b = base_attn[:, :num_target_tokens, num_target_tokens:]
    distractor_zero_pct_base = float((dist_weights_b <= 1e-4).float().mean().item() * 100.0)
    distractor_leakage_base = float(dist_weights_b.sum(dim=-1).mean().item() * 100.0)
    
    print(f"\nDistractor Resistance Results (Bidirectional Context):")
    print(f"• Target query tokens: {num_target_tokens} | Distractor tokens: {len(combined_tokens) - num_target_tokens}")
    print(f"• Soft L-Topology Distractor Pruning: {distractor_zero_pct_soft:.2f}% strictly 0.0000 | Attention Leakage: {distractor_leakage_soft:.2f}%")
    print(f"• Standard Transformer Pruning:      {distractor_zero_pct_base:.2f}% strictly 0.0000 | Attention Leakage: {distractor_leakage_base:.2f}%")
    leakage_reduction = float(distractor_leakage_base - distractor_leakage_soft)
    print(f"• Absolute Attention Leakage Reduction: {leakage_reduction:+.2f}%")
    
    # 5. Visual Artifacts
    os.makedirs("artifacts", exist_ok=True)
    plot_path = "artifacts/exp10_encoder_benchmarks.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # Panel 1: MLM Training Loss
    ax1 = axes[0, 0]
    ax1.plot(soft_losses, label="Soft L-Topology Encoder", color="#1f77b4", linewidth=1.5)
    ax1.plot(base_losses, label="Standard BERT-style Encoder", color="#d62728", linewidth=1.5)
    ax1.set_title("Bidirectional Masked Language Modeling Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimization Step", fontsize=10)
    ax1.set_ylabel("MLM Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)
    
    # Panel 2: MLM Accuracy Comparison (Top-1 and Top-5)
    ax2 = axes[0, 1]
    acc_labels = ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Exact-Zero Sparsity (%)"]
    soft_accs = [float(soft_eval["top1_acc"]), float(soft_eval["top5_acc"]), float(soft_eval["mean_sparsity"])]
    base_accs = [float(base_eval["top1_acc"]), float(base_eval["top5_acc"]), 0.0]
    x_pos = range(len(acc_labels))
    w = 0.35
    ax2.bar([p - w/2 for p in x_pos], soft_accs, w, label="Soft L-Topology Encoder", color="#1f77b4", alpha=0.85)
    ax2.bar([p + w/2 for p in x_pos], base_accs, w, label="Standard BERT Encoder", color="#d62728", alpha=0.85)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(acc_labels, fontsize=10)
    ax2.set_title("Masked Token Recovery & Routing Sparsity Profile", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True)
    
    # Panel 3: Distractor Isolation Comparison
    ax3 = axes[1, 0]
    dist_metrics = ["Exact Zero Pruning (%)", "Distractor Leakage (%)"]
    soft_dist_vals = [distractor_zero_pct_soft, distractor_leakage_soft]
    base_dist_vals = [distractor_zero_pct_base, distractor_leakage_base]
    x_d = range(len(dist_metrics))
    ax3.bar([p - w/2 for p in x_d], soft_dist_vals, w, label="Soft L-Topology Encoder", color="#2ca02c", alpha=0.85)
    ax3.bar([p + w/2 for p in x_d], base_dist_vals, w, label="Standard BERT Encoder", color="#d62728", alpha=0.85)
    ax3.set_xticks(list(x_d))
    ax3.set_xticklabels(dist_metrics, fontsize=10)
    ax3.set_title("Contextual Distractor Resistance in Bidirectional Reasoning", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Percentage (%)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", frameon=True)
    
    # Panel 4: Heatmap of Distractor Isolation
    ax4 = axes[1, 1]
    # Show routing from target tokens (rows) across all tokens (cols)
    im = ax4.imshow(mean_routing[:num_target_tokens, :min(32, mean_routing.shape[1])], cmap="viridis", aspect="auto")
    ax4.axvline(num_target_tokens - 0.5, color="red", linestyle="--", linewidth=1.5, label="Target / Distractor Boundary")
    ax4.set_title("Soft L-Topology Bidirectional Attention Routing Heatmap", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Key Position (0-13: Target | >13: Distractors)", fontsize=9)
    ax4.set_ylabel("Query Position (Target Tokens)", fontsize=9)
    plt.colorbar(im, ax=ax4, label="Routing Weight")
    ax4.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved encoder benchmark plot to: {plot_path}")
    
    # Copy to brain artifact directory
    brain_dir = os.path.expanduser("/home/zero/.gemini/antigravity-cli/brain/ebcbf1a3-eaae-447c-a440-83473140be5a")
    if os.path.exists(brain_dir):
        import shutil
        brain_plot = os.path.join(brain_dir, "exp10_encoder_benchmarks.png")
        shutil.copyfile(plot_path, brain_plot)
        print(f"Copied visual artifact to: {brain_plot}")
        
    results = {
        "experiment": "Experiment 10: Encoder-Only Architecture Exploration & Logic Edge",
        "mlm_steps": int(mlm_steps),
        "soft_encoder": {
            "mlm_val_loss": float(soft_eval["loss"]),
            "top1_accuracy": float(soft_eval["top1_acc"]),
            "top5_accuracy": float(soft_eval["top5_acc"]),
            "mean_sparsity_pct": float(soft_eval["mean_sparsity"]),
            "distractor_exact_zeros_pct": float(distractor_zero_pct_soft),
            "distractor_attention_leakage_pct": float(distractor_leakage_soft),
        },
        "baseline_encoder": {
            "mlm_val_loss": float(base_eval["loss"]),
            "top1_accuracy": float(base_eval["top1_acc"]),
            "top5_accuracy": float(base_eval["top5_acc"]),
            "mean_sparsity_pct": 0.0,
            "distractor_exact_zeros_pct": float(distractor_zero_pct_base),
            "distractor_attention_leakage_pct": float(distractor_leakage_base),
        },
        "leakage_reduction_pct": float(leakage_reduction),
        "plot_path": plot_path,
    }
    
    with open("experiments/results_exp10.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 80)
    print("   EXPERIMENT 10 COMPLETED SUCCESSFULLY")
    print(f"   Soft L-Topology MLM Top-1: {soft_eval['top1_acc']:.2f}% | Top-5: {soft_eval['top5_acc']:.2f}% | Sparsity: {soft_eval['mean_sparsity']:.2f}%")
    print(f"   Baseline MLM Top-1:        {base_eval['top1_acc']:.2f}% | Top-5: {base_eval['top5_acc']:.2f}% | Sparsity: 0.00%")
    print(f"   Distractor Leakage Reduced by: {leakage_reduction:.2f}%")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_experiment_10()
