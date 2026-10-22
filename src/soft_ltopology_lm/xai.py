"""
Explainable AI (XAI) and Sparsity Analysis Utilities.
Visualizes soft L-topological routing heatmaps, plots comparative sparsity charts,
and logs deterministic mathematical traces of token-level lattice intersections.
"""

import os
import torch
from typing import List, Optional, Dict, Union, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def log_deterministic_reasoning_trace(
    tokens: List[str],
    routing_trace: Union[torch.Tensor, Dict[str, torch.Tensor]],
    layer_idx: int = 0,
    head_idx: int = 0,
    threshold: float = 1e-5,
    trace_type: str = "norm_routing",
) -> str:
    r"""
    Generates a textual mathematical trace of token intersections:
    Showing exactly which tokens had non-zero containment and which had dead / exact-zero logic.
    """
    if isinstance(routing_trace, dict):
        mat = routing_trace.get(trace_type, routing_trace.get("norm_routing", routing_trace.get("containment")))
    else:
        mat = routing_trace
        
    if mat.ndim == 4:
        mat = mat[0, head_idx].detach().cpu()
    elif mat.ndim == 3:
        mat = mat[head_idx].detach().cpu()
    else:
        mat = mat.detach().cpu()
        
    seq_len = len(tokens)
    lines = []
    lines.append(f"=== XAI Deterministic Lattice Trace (Layer {layer_idx}, Head {head_idx}) ===")
    
    for i in range(seq_len):
        target_tok = repr(tokens[i])
        active_sources = []
        zero_count = 0
        for j in range(i + 1):
            weight = mat[i, j].item()
            if weight > threshold:
                active_sources.append(f"{repr(tokens[j])}: {weight:.4f}")
            else:
                zero_count += 1
                
        sparsity_row = (zero_count / (i + 1)) * 100.0
        sources_str = ", ".join(active_sources) if active_sources else "None (Independent Token)"
        lines.append(f"Token [{i:2d}] {target_tok:<15} <- Intersections: [{sources_str}] (Sparsity: {sparsity_row:5.1f}% zeros)")
        
    return "\n".join(lines)


def save_routing_heatmap(
    tokens: List[str],
    routing_trace: Union[torch.Tensor, Dict[str, torch.Tensor]],
    save_path: str = "artifacts/lattice_routing_heatmap.png",
    layer_idx: int = 0,
    head_idx: int = 0,
    title_suffix: str = "Lattice Routing Distribution",
    trace_type: str = "norm_routing",
) -> str:
    r"""
    Renders and saves an annotated heatmap of token intersections showing exact zero sparsity.
    """
    if isinstance(routing_trace, dict):
        mat = routing_trace.get(trace_type, routing_trace.get("norm_routing", routing_trace.get("containment")))
    else:
        mat = routing_trace
        
    if mat.ndim == 4:
        mat = mat[0, head_idx].detach().cpu().numpy()
    elif mat.ndim == 3:
        mat = mat[head_idx].detach().cpu().numpy()
    else:
        mat = mat.detach().cpu().numpy()
        
    S = len(tokens)
    mat = mat[:S, :S]
    
    tril_mask = np.tril(np.ones((S, S), dtype=bool))
    valid_vals = mat[tril_mask]
    zero_pct = (np.abs(valid_vals) <= 1e-6).sum() / max(len(valid_vals), 1) * 100.0

    fig, ax = plt.subplots(figsize=(max(8, S * 0.7), max(6, S * 0.6)), dpi=150)
    
    cmap = plt.cm.viridis
    cax = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=max(float(mat.max()), 1e-3))
    
    ax.set_xticks(range(S))
    ax.set_yticks(range(S))
    ax.set_xticklabels([repr(t) for t in tokens], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([repr(t) for t in tokens], fontsize=9)
    
    ax.set_xlabel("Key Context Tokens", fontsize=11, fontweight="bold")
    ax.set_ylabel("Query Tokens", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Soft L-Topological Sequence Routing (Layer {layer_idx}, Head {head_idx})\n"
        f"{title_suffix} | Causal Exact-Zero Sparsity: {zero_pct:.1f}%",
        fontsize=12, fontweight="bold", pad=12
    )
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Lattice Containment Weight", fontsize=10)
    
    if S <= 16:
        for i in range(S):
            for j in range(i + 1):
                val = mat[i, j]
                text_col = "white" if val < (mat.max() * 0.5) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=7)
                
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def save_multihead_comparison_heatmap(
    tokens: List[str],
    traces: List[Dict[str, torch.Tensor]],
    save_path: str = "artifacts/xai_multihead_heatmap.png",
    layer_idx: int = 0,
) -> str:
    r"""
    Generates a 3-panel figure demonstrating multi-scale logical routing across layers and heads.
    """
    S = len(tokens)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    
    configs = [
        (0, 0, "Layer 0, Head 0 (Contextual Inclusion)"),
        (min(2, len(traces)-1), 2, f"Layer {min(2, len(traces)-1)}, Head 2 (Deterministic Sparse Logic)"),
        (min(3, len(traces)-1), 0, f"Layer {min(3, len(traces)-1)}, Head 0 (Hierarchical Semantic Routing)"),
    ]
    
    tril_mask = np.tril(np.ones((S, S), dtype=bool))
    
    for idx, (l_idx, h_idx, title) in enumerate(configs):
        ax = axes[idx]
        norm_routing = traces[l_idx]["norm_routing"][0].detach().cpu().numpy()
        mat = norm_routing[h_idx, :S, :S]
        valid_vals = mat[tril_mask]
        sparsity_pct = (np.abs(valid_vals) <= 1e-6).sum() / max(len(valid_vals), 1) * 100.0
        
        cax = ax.imshow(mat, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
        ax.set_xticklabels([repr(t) for t in tokens], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([repr(t) for t in tokens], fontsize=8)
        ax.set_title(f"{title}\nExact-Zero Sparsity: {sparsity_pct:.1f}%", fontsize=10, fontweight="bold")
        ax.set_xlabel("Key Context Tokens", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Query Tokens", fontsize=9)
            
        if S <= 12:
            for i in range(S):
                for j in range(i + 1):
                    val = mat[i, j]
                    col = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=col, fontsize=7)
                    
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
    plt.suptitle("Soft L-Topological Hierarchical Explainability (XAI Trace Across Layers & Heads)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_sparsity_comparison(
    soft_l_sparsity: float,
    baseline_sparsity: float = 0.0,
    save_path: str = "artifacts/sparsity_comparison.png",
) -> str:
    r"""
    Generates a comparative bar plot comparing exact zero sparsity of Soft L-Topology vs Transformer.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    
    categories = ["Standard Transformer\n(Dot-Product + Softmax)", "Soft L-Topological LM\n(Continuous Lattice Router)"]
    sparsities = [baseline_sparsity, soft_l_sparsity]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = ax.bar(categories, sparsities, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Exact Zero Routing Sparsity (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("Deterministic Routing Sparsity Evaluation\nDead Connections Evaluating to Strictly 0.0", fontsize=12, fontweight="bold")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )
        
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def compare_sparsity_summary(
    soft_l_sparsity: float,
    baseline_sparsity: float = 0.0,
) -> str:
    r"""
    Summarizes routing sparsity comparison against standard softmax Transformer.
    """
    summary = (
        f"\n================ ROUTING SPARSITY EVALUATION ================\n"
        f"Soft L-Topological Router (Exact Zeros): {soft_l_sparsity:6.2f}%\n"
        f"Standard Softmax Attention (Exact Zeros): {baseline_sparsity:6.2f}%\n"
        f"-> Mathematical True Sparsity Gain:       +{soft_l_sparsity - baseline_sparsity:6.2f}%\n"
        f"=============================================================\n"
    )
    return summary


def format_autoregressive_trace(
    prompt: str,
    generated_text: str,
    step_logs: List[Dict[str, Any]],
) -> str:
    r"""
    Formats step-by-step autoregressive decisions into a clear, structured explanation.
    """
    lines = []
    lines.append("\n" + "=" * 82)
    lines.append("        EXPLAINABLE AUTOREGRESSIVE GENERATION TRACE (STEP-BY-STEP)")
    lines.append("        Soft L-Topological Continuous Lattice Sequence Routing")
    lines.append("=" * 82)
    lines.append(f"Prompt: {repr(prompt)}")
    lines.append("=" * 82)
    
    for log in step_logs:
        step = log["step"]
        chosen = repr(log["chosen_token"])
        prob = log["chosen_prob"] * 100.0
        
        q_sparsity = log.get("query_sparsity_pct", log.get("sparsity_pct", 0.0))
        dead_p = log.get("dead_paths", None)
        tot_p = log.get("total_paths", None)
        if dead_p is not None and tot_p is not None:
            sparsity_str = f"{q_sparsity:5.1f}% zeros ({dead_p}/{tot_p} dead paths strictly 0.0)"
        else:
            sparsity_str = f"{q_sparsity:5.1f}% zeros"
            
        cands = log.get("candidates", [])
        cands_str = " | ".join([f"{repr(c[0])} ({c[1]*100:.1f}%)" for c in cands])
        
        active = log.get("active_sources", [])
        if active and isinstance(active[0], dict):
            active_sorted = sorted(active, key=lambda x: x["max_weight"], reverse=True)[:4]
            active_str = ", ".join([f"{repr(a['token'])} (max {a['max_weight']*100:.1f}%, H{a['primary_head']})" for a in active_sorted])
        elif active and isinstance(active[0], (list, tuple)):
            active_sorted = sorted(active, key=lambda x: x[1], reverse=True)[:4]
            active_str = ", ".join([f"{repr(a[0])} ({a[1]*100:.1f}%)" for a in active_sorted])
        else:
            active_str = "Autonomous Start (No Prior Dependency)"
            
        pruned = log.get("pruned_sources", [])
        if pruned:
            pruned_unique = list(dict.fromkeys(pruned))[:5]
            pruned_str = ", ".join([repr(p) for p in pruned_unique])
            if len(pruned) > 5:
                pruned_str += f" (+{len(pruned)-5} more)"
        else:
            pruned_str = "None (All tokens attended by at least 1 head)"
            
        head_det = log.get("head_details", [])
        if head_det:
            head_summaries = []
            for hd in head_det:
                h_id = hd["head"]
                h_sp = hd["sparsity_pct"]
                h_top = repr(hd["top_token"])
                head_summaries.append(f"H{h_id}: {h_sp:.0f}% zeros (top: {h_top})")
            head_str = " | ".join(head_summaries)
        else:
            head_str = ""

        lines.append(f"┌─ Step {step:02d}: Chosen Token {chosen:<12} (Confidence: {prob:5.1f}%)")
        lines.append(f"│  • Considered Candidates: [{cands_str}]")
        lines.append(f"│  • Token Path Sparsity:   {sparsity_str}")
        lines.append(f"│  • Active Antecedents:    [{active_str}]")
        lines.append(f"│  • Pruned Tokens (0.0):   [{pruned_str}]")
        if head_str:
            lines.append(f"│  • Head Specialization:   [{head_str}]")
        lines.append(f"└─────────────────────────────────────────────────────────────────────────────\n")

    lines.append("-" * 82)
    lines.append(f"Complete Autoregressive Continuation:\n{generated_text}")
    lines.append("=" * 82 + "\n")
    return "\n".join(lines)


def save_autoregressive_step_heatmap(
    prompt: str,
    step_logs: List[Dict[str, Any]],
    save_path: str = "artifacts/bpe_autoregressive_trace.png",
) -> str:
    r"""
    Generates and saves a publication-quality heatmap showing dynamic routing weights
    for each autoregressive generation step.
    """
    if not step_logs:
        return save_path
        
    num_steps = len(step_logs)
    last_log = step_logs[-1]
    all_context_tokens = last_log.get("all_cond_tokens", [])
    if not all_context_tokens:
        return save_path
        
    total_ctx_len = len(all_context_tokens)
    matrix = np.full((num_steps, total_ctx_len), np.nan)
    
    for i, log in enumerate(step_logs):
        row = log.get("mean_routing_row", None)
        if row is not None:
            ctx_len = len(row)
            matrix[i, :ctx_len] = row
            
    fig, ax = plt.subplots(figsize=(max(8, total_ctx_len * 0.75), max(5, num_steps * 0.6)), dpi=150)
    
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f4f6f7")
    
    masked_matrix = np.ma.masked_invalid(matrix)
    vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) and np.nanmax(matrix) > 0 else 1.0
    cax = ax.imshow(masked_matrix, cmap=cmap, vmin=0.0, vmax=vmax)
    
    gen_labels = [f"Step {log['step']}: {repr(log['chosen_token'])}" for log in step_logs]
    ctx_labels = [repr(t) for t in all_context_tokens]
    
    ax.set_xticks(range(total_ctx_len))
    ax.set_yticks(range(num_steps))
    ax.set_xticklabels(ctx_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(gen_labels, fontsize=8)
    ax.set_xlabel("Context Tokens (Prompt + Decoded History)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Autoregressive Generation Steps", fontsize=9, fontweight="bold")
    
    ax.set_title(
        f"Dynamic Autoregressive Lattice Routing Trace\nPrompt: {repr(prompt)} (Exact 0.0 = Dead Connections / Non-Hallucination)",
        fontsize=10,
        fontweight="bold",
        pad=10
    )
    
    for i in range(num_steps):
        row_len = len(step_logs[i].get("mean_routing_row", []))
        for j in range(row_len):
            val = matrix[i, j]
            if not np.isnan(val):
                col = "white" if val > 0.35 * vmax else "black"
                txt = "0.0" if val <= 1e-4 else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=7)
                
    fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
