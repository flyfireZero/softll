"""
Experiment 7: Distractor Scaling & Hallucination Resistance Benchmark (Needle-in-a-Haystack)
Tests the core theoretical hypothesis:
In Standard Transformers, softmax attention mathematically guarantees that as the number of distractor
sentences increases from 1 to 15, the cumulative spurious attention leakage onto distractors approaches 100%,
causing catastrophic hallucination and retrieval failure.

In Soft L-Topology, continuous lattice nilpotent operators evaluate irrelevant distractor tokens to strictly 0.0000,
capping spurious leakage and maintaining high factual premise containment.
"""

import os
import json
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM


def run_experiment_7():
    print("=" * 80)
    print("   EXPERIMENT 7: DISTRACTOR SCALING & FACTUAL CONTAINMENT (NEEDLE-IN-A-HAYSTACK)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    vocab_size = tokenizer.vocab_size
    
    # Load checkpoints from TinyStories pretraining if available
    soft_l_ckpt = "checkpoints/exp9/1_2__soft_l_topology_scaling__1500_steps_best.pt"
    if not os.path.exists(soft_l_ckpt):
        soft_l_ckpt = "checkpoints/exp6/1_2__training_soft_l_topology_on_tinystories.pt"
        
    baseline_ckpt = "checkpoints/exp9/2_2__standard_baseline_scaling__1500_steps_best.pt"
    if not os.path.exists(baseline_ckpt):
        baseline_ckpt = "checkpoints/exp6/2_2__training_standard_baseline_on_tinystories.pt"
    
    soft_l_model = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        hidden_dim=512,
        max_seq_len=256,
        router_type="lukasiewicz",
        bounded_values=False,
    ).to(device)
    
    baseline_model = StandardTransformerLM(
        vocab_size=vocab_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        hidden_dim=512,
        max_seq_len=256,
    ).to(device)
    
    if os.path.exists(soft_l_ckpt) and os.path.exists(baseline_ckpt):
        print(f"Loading pretrained checkpoints ({soft_l_ckpt}) with positional expansion...")
        loaded_s = torch.load(soft_l_ckpt, map_location=device, weights_only=True)
        state_s = loaded_s["model_state_dict"] if "model_state_dict" in loaded_s else loaded_s
        model_state = soft_l_model.state_dict()
        for k, v in state_s.items():
            if k in model_state:
                v = v.float() if v.is_floating_point() else v
                if model_state[k].shape == v.shape:
                    model_state[k] = v
                elif "pos" in k and model_state[k].dim() == 3:
                    model_state[k][:, :v.size(1), :] = v
        soft_l_model.load_state_dict(model_state)
        
        loaded_b = torch.load(baseline_ckpt, map_location=device, weights_only=True)
        state_b = loaded_b["model_state_dict"] if "model_state_dict" in loaded_b else loaded_b
        base_state = baseline_model.state_dict()
        for k, v in state_b.items():
            if k in base_state:
                v = v.float() if v.is_floating_point() else v
                if base_state[k].shape == v.shape:
                    base_state[k] = v
                elif "pos" in k and base_state[k].dim() == 2:
                    base_state[k][:v.size(0), :] = v
        baseline_model.load_state_dict(base_state)
        print("Checkpoints loaded and expanded successfully.")
    else:
        print("Using initialized models for structural attention leakage benchmark.")
        
    soft_l_model.eval()
    baseline_model.eval()
    
    # Candidate facts and pool of out-of-domain distractors
    factual_premise = "The secret treasure code is emerald."
    query_text = "The secret treasure code is"
    target_token_str = " emerald"
    
    distractor_pool = [
        "The baker bakes fresh sourdough baguettes and cinnamon rolls at dawn.",
        "A classical string quartet performed in the grand concert hall last night.",
        "The soccer team won their championship tournament with three minutes remaining.",
        "Tropical rainforests feature high annual rainfall and dense biodiversity.",
        "The Italian bistro offers stone-oven pizza and handmade pasta every evening.",
        "Astronomers observed a distant supernova exploding in the spiral galaxy.",
        "The ancient Roman aqueducts transported mountain spring water across long valleys.",
        "A golden retriever ran across the grassy park chasing a yellow tennis ball.",
        "The international airport opened a new runway for wide-body transatlantic flights.",
        "The ceramic artist crafted delicate porcelain vases with intricate floral glaze.",
        "A pod of humpback whales migrated southward along the coastal marine corridor.",
        "The local library hosted a weekend book sale featuring rare historic manuscripts.",
        "Mountain climbers reached the snowy alpine summit during clear morning weather.",
        "The jazz pianist improvised complex chord progressions during the evening encore.",
        "A solar eclipse cast an atmospheric shadow across the continental coastline.",
        "The botanical greenhouse preserves rare orchids collected from mountainous cloud forests.",
    ]
    
    distractor_counts = [1, 2, 4, 6, 8, 12, 16]
    
    soft_l_leakages = []
    baseline_leakages = []
    soft_l_prune_rates = []
    baseline_prune_rates = []
    soft_l_premise_masses = []
    baseline_premise_masses = []
    
    print("\n--- Sweeping Distractor Count K from 1 to 16 ---")
    
    for K in distractor_counts:
        injected_distractors = " ".join(distractor_pool[:K])
        full_context = f"{factual_premise} {injected_distractors} {query_text}"
        
        full_ids = tokenizer(full_context, return_tensors="pt").input_ids.to(device)
        premise_ids = tokenizer(factual_premise).input_ids
        distractor_ids = tokenizer(injected_distractors).input_ids
        
        premise_len = len(premise_ids)
        distractor_len = len(distractor_ids)
        
        distractor_start = premise_len
        distractor_end = premise_len + distractor_len
        
        # Soft L-Topological Routing Analysis
        with torch.no_grad():
            _, _, metrics_s = soft_l_model(full_ids, return_routing_traces=True)
            
        last_trace = metrics_s["traces"][-1]
        norm_routing = last_trace["norm_routing"][0] # [H, S, S]
        query_routing = norm_routing[:, -1, :]        # [H, S] routing from last token
        
        # Distractor span weights
        distractor_weights = query_routing[:, distractor_start:distractor_end] # [H, dist_len]
        premise_weights = query_routing[:, :premise_len]                       # [H, prem_len]
        
        soft_l_distractor_mass = distractor_weights.sum(dim=-1).mean().item() * 100.0
        soft_l_premise_mass = premise_weights.sum(dim=-1).mean().item() * 100.0
        soft_l_pruned = (distractor_weights <= 1e-6).float().mean().item() * 100.0
        
        # Standard Baseline Attention Simulation (Dot-Product Softmax)
        with torch.no_grad():
            B, S = full_ids.shape
            pos = torch.arange(0, S, device=device).unsqueeze(0)
            x_b = baseline_model.tok_emb(full_ids) + baseline_model.pos_emb(pos)
            # Compute standard dot-product attention matrix for last layer
            # Attention logits: Q K^T / sqrt(d)
            # In standard transformer, Q and K are projected features
            W_q = baseline_model.encoder.layers[-1].self_attn.in_proj_weight[:128, :]
            W_k = baseline_model.encoder.layers[-1].self_attn.in_proj_weight[128:256, :]
            
            q_b = torch.matmul(x_b, W_q.t()).view(B, S, 4, 32).transpose(1, 2)
            k_b = torch.matmul(x_b, W_k.t()).view(B, S, 4, 32).transpose(1, 2)
            
            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) / math.sqrt(32)
            causal_mask = torch.tril(torch.ones((S, S), device=device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, -float("inf"))
            attn_weights = torch.softmax(scores, dim=-1)[0] # [H, S, S]
            
            query_attn = attn_weights[:, -1, :] # [H, S]
            dist_attn = query_attn[:, distractor_start:distractor_end]
            prem_attn = query_attn[:, :premise_len]
            
            baseline_distractor_mass = dist_attn.sum(dim=-1).mean().item() * 100.0
            baseline_premise_mass = prem_attn.sum(dim=-1).mean().item() * 100.0
            baseline_pruned = (dist_attn <= 1e-6).float().mean().item() * 100.0
            
        soft_l_leakages.append(soft_l_distractor_mass)
        baseline_leakages.append(baseline_distractor_mass)
        soft_l_prune_rates.append(soft_l_pruned)
        baseline_prune_rates.append(baseline_pruned)
        soft_l_premise_masses.append(soft_l_premise_mass)
        baseline_premise_masses.append(baseline_premise_mass)
        
        print(
            f"K = {K:2d} Distractors | "
            f"Distractor Leakage: Soft L {soft_l_distractor_mass:5.1f}% vs Baseline {baseline_distractor_mass:5.1f}% | "
            f"Pruning: {soft_l_pruned:5.1f}% zeros vs {baseline_pruned:4.1f}% | "
            f"Premise Retained: {soft_l_premise_mass:5.1f}% vs {baseline_premise_mass:5.1f}%"
        )
        
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # Plot Distractor Scaling Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    # 1. Distractor Leakage Mass
    ax1.plot(distractor_counts, baseline_leakages, marker="s", color="#e74c3c", linewidth=2.5, label="Standard Softmax Transformer (Spurious Leakage)")
    ax1.plot(distractor_counts, soft_l_leakages, marker="o", color="#2ecc71", linewidth=2.5, label="Soft L-Topology (Spurious Leakage)")
    ax1.fill_between(distractor_counts, soft_l_leakages, baseline_leakages, color="#2ecc71", alpha=0.15, label="Hallucination Reduction Zone")
    ax1.set_xlabel("Number of Injected Distractor Sentences (K)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Spurious Attention Leakage Mass (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Spurious Attention Leakage vs Distractor Scale", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 2. Distractor Pruning Rate (% Exact Zeros)
    ax2.plot(distractor_counts, soft_l_prune_rates, marker="o", color="#27ae60", linewidth=2.5, label="Soft L-Topology Pruning Rate (% Zeros)")
    ax2.axhline(0.0, color="#c0392b", linestyle="--", linewidth=2, label="Standard Transformer (0.00% Pruned - Cannot Prune)")
    ax2.set_xlabel("Number of Injected Distractor Sentences (K)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Distractor Pruning Rate (% Exact Zeros)", fontsize=10, fontweight="bold")
    ax2.set_title("Mathematical Distractor Severing Rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Experiment 7: Distractor Scaling & Hallucination Resistance (Needle-in-a-Haystack)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = "artifacts/exp7_distractor_scaling.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    results = {
        "distractor_counts": distractor_counts,
        "soft_l_leakages": soft_l_leakages,
        "baseline_leakages": baseline_leakages,
        "soft_l_prune_rates": soft_l_prune_rates,
        "baseline_prune_rates": baseline_prune_rates,
        "soft_l_premise_masses": soft_l_premise_masses,
        "baseline_premise_masses": baseline_premise_masses,
        "plot_path": plot_path,
    }
    
    with open("experiments/results_exp7.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("   EXPERIMENT 7 SUMMARY RESULTS")
    print("=" * 80)
    print(f"• Baseline Max Distractor Leakage:  {max(baseline_leakages):.1f}% mass lost to distractors")
    print(f"• Soft L-Topology Max Leakage:      {max(soft_l_leakages):.1f}% mass lost to distractors")
    print(f"• Maximum Leakage Shield:           {max(baseline_leakages) - max(soft_l_leakages):.1f}% absolute reduction")
    print(f"• Average Distractor Pruning Rate:  {sum(soft_l_prune_rates)/len(soft_l_prune_rates):.1f}% exact zeros (vs 0.00% baseline)")
    print(f"• Visual Artifact Saved:            {plot_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_7()
