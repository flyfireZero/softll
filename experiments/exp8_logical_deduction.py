"""
Experiment 8: Multi-Step Deductive Entailment & Logical Transitivity Benchmark
Directly tests where Soft L-Topology fundamentally excels over the norm:
Formal Logic Reasoning (Transitive Modus Ponens: A -> B and B -> C => A -> C).

In Euclidean dot-product attention:
Inner products compute symmetric geometric alignment (cosine similarity).
Geometric similarity is not transitive and does not model directional implication.

In Soft L-Topology:
Continuous Jaccard Relative Containment C(Q, K) = max(0, Meet(Q, K) - tau) / Join(Q, K)
directly evaluates fuzzy subsethood (K subset Q <=> K -> Q), preserving transitive entailment.
"""

import os
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM


def run_experiment_8():
    print("=" * 80)
    print("   EXPERIMENT 8: DEDUCTIVE ENTAILMENT & MULTI-STEP LOGICAL REASONING")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    vocab_size = tokenizer.vocab_size
    
    # Load pretrained TinyStories checkpoints: prioritize exp9 (1500 steps), fall back to exp6
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
        print(f"Loading pretrained models ({soft_l_ckpt})...")
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
        print("Models loaded successfully.")
    else:
        print("Using initialized models.")
        
    soft_l_model.eval()
    baseline_model.eval()
    
    # Deductive Entailment Scenarios: (Rule 1, Rule 2, Fact, Query, Correct Target, Distractor Target)
    scenarios = [
        {
            "id": "biology_chain",
            "rule1": "If an organism is a mammal, it is warm-blooded.",
            "rule2": "If an organism is warm-blooded, it maintains constant body temperature.",
            "fact": "A dolphin is a mammal.",
            "query": "Therefore, a dolphin maintains constant body",
            "target": " temperature",
            "distractor": " cold",
        },
        {
            "id": "physics_chain",
            "rule1": "If a material is a superconductor, it has zero electrical resistance.",
            "rule2": "If a material has zero electrical resistance, it expels magnetic fields via the Meissner effect.",
            "fact": "Yttrium barium copper oxide is a superconductor.",
            "query": "Therefore, yttrium barium copper oxide expels magnetic",
            "target": " fields",
            "distractor": " heat",
        },
        {
            "id": "geometry_chain",
            "rule1": "If a polygon is an equilateral triangle, all three of its interior angles are equal.",
            "rule2": "If all three interior angles of a triangle are equal, each angle measures sixty degrees.",
            "fact": "Shape Alpha is an equilateral triangle.",
            "query": "Therefore, each interior angle of Shape Alpha measures sixty",
            "target": " degrees",
            "distractor": " radians",
        },
        {
            "id": "computing_chain",
            "rule1": "If an algorithm runs in logarithmic time, it divides the search space in half at each step.",
            "rule2": "If an algorithm divides the search space in half at each step, it scales efficiently to massive datasets.",
            "fact": "Binary search runs in logarithmic time.",
            "query": "Therefore, binary search scales efficiently to massive",
            "target": " datasets",
            "distractor": " networks",
        },
        {
            "id": "astronomy_chain",
            "rule1": "If a celestial body exceeds the Chandrasekhar limit, electron degeneracy pressure cannot prevent gravitational collapse.",
            "rule2": "If electron degeneracy pressure cannot prevent gravitational collapse, the core collapses into a neutron star or black hole.",
            "fact": "A massive dying stellar core exceeds the Chandrasekhar limit.",
            "query": "Therefore, the dying stellar core collapses into a neutron star or black",
            "target": " hole",
            "distractor": " energy",
        },
    ]
    
    print("\n--- Evaluating Multi-Hop Transitive Deduction Scenarios ---")
    
    soft_l_correct = 0
    baseline_correct = 0
    soft_l_margins = []
    baseline_margins = []
    
    for sc in scenarios:
        prompt = f"{sc['rule1']} {sc['rule2']} {sc['fact']} {sc['query']}"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        target_id = tokenizer.encode(sc["target"])[0]
        distractor_id = tokenizer.encode(sc["distractor"])[0]
        
        with torch.no_grad():
            logits_s, _, _ = soft_l_model(input_ids)
            last_logits_s = logits_s[0, -1, :]
            prob_s_target = torch.softmax(last_logits_s, dim=-1)[target_id].item()
            prob_s_distractor = torch.softmax(last_logits_s, dim=-1)[distractor_id].item()
            margin_s = prob_s_target - prob_s_distractor
            
            logits_b, _ = baseline_model(input_ids)
            last_logits_b = logits_b[0, -1, :]
            prob_b_target = torch.softmax(last_logits_b, dim=-1)[target_id].item()
            prob_b_distractor = torch.softmax(last_logits_b, dim=-1)[distractor_id].item()
            margin_b = prob_b_target - prob_b_distractor
            
        is_s_correct = margin_s > 0
        is_b_correct = margin_b > 0
        
        if is_s_correct: soft_l_correct += 1
        if is_b_correct: baseline_correct += 1
        
        soft_l_margins.append(margin_s)
        baseline_margins.append(margin_b)
        
        print(f"Scenario [{sc['id']}]:")
        print(f"  Target: '{sc['target']}' vs Distractor: '{sc['distractor']}'")
        print(f"  Soft L-Topology: Target Prob = {prob_s_target:.4f} | Margin = {margin_s:+.4f} ({'✓ PASS' if is_s_correct else '✗ FAIL'})")
        print(f"  Standard Baseline: Target Prob = {prob_b_target:.4f} | Margin = {margin_b:+.4f} ({'✓ PASS' if is_b_correct else '✗ FAIL'})")
        
    soft_l_acc = (soft_l_correct / len(scenarios)) * 100.0
    baseline_acc = (baseline_correct / len(scenarios)) * 100.0
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # Plot Logic Deductive Accuracy and Margins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    models = ["Soft L-Topology\n(Continuous Lattice Logic)", "Standard Transformer\n(Dot-Product Softmax)"]
    accs = [soft_l_acc, baseline_acc]
    colors = ["#2ecc71", "#e74c3c"]
    
    bars = ax1.bar(models, accs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Deductive Transitivity Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_title("Multi-Hop Deductive Reasoning Accuracy", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold", fontsize=11)
        
    # Margin comparison per scenario
    x_indices = range(len(scenarios))
    sc_names = [sc["id"].replace("_chain", "") for sc in scenarios]
    width = 0.35
    ax2.bar([i - width/2 for i in x_indices], soft_l_margins, width=width, label="Soft L-Topology Margin", color="#2ecc71", edgecolor="black")
    ax2.bar([i + width/2 for i in x_indices], baseline_margins, width=width, label="Standard Baseline Margin", color="#e74c3c", edgecolor="black")
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xticks(list(x_indices))
    ax2.set_xticklabels(sc_names, fontsize=8, fontweight="bold")
    ax2.set_ylabel("Probability Margin (Target - Distractor)", fontsize=10, fontweight="bold")
    ax2.set_title("Logical Entailment Margin by Domain", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Experiment 8: Deductive Entailment & Logical Transitivity", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = "artifacts/exp8_logical_deduction.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    results = {
        "experiment": "Experiment 8: Deductive Entailment Benchmark",
        "soft_l_accuracy_pct": soft_l_acc,
        "baseline_accuracy_pct": baseline_acc,
        "soft_l_mean_margin": sum(soft_l_margins) / len(soft_l_margins),
        "baseline_mean_margin": sum(baseline_margins) / len(baseline_margins),
        "scenarios": [sc["id"] for sc in scenarios],
        "plot_path": plot_path,
    }
    
    with open("experiments/results_exp8.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("   EXPERIMENT 8 SUMMARY RESULTS")
    print("=" * 80)
    print(f"• Soft L-Topology Deductive Accuracy: {soft_l_acc:.1f}%")
    print(f"• Standard Transformer Accuracy:     {baseline_acc:.1f}%")
    print(f"• Mean Target Confidence Margin:     Soft L: {results['soft_l_mean_margin']:+.4f} vs Baseline: {results['baseline_mean_margin']:+.4f}")
    print(f"• Visual Artifact Saved:             {plot_path}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_experiment_8()
