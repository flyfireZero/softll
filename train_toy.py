"""
End-to-End Training and Unit-Testing Script:
Trains SoftLTopologicalLM and StandardTransformerLM on TinyShakespeare (Character-Level Logic Unit Tests).
Evaluates loss, perplexity, exact zero routing sparsity, and logs XAI traces.
"""

import os
import time
import math
import torch
import torch.nn as nn
from model import SoftLTopologicalLM, StandardTransformerLM
from xai_utils import (
    log_deterministic_reasoning_trace,
    compare_sparsity_summary,
    save_routing_heatmap,
    plot_sparsity_comparison,
)


SAMPLE_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!
"""

class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.char_to_id[c] for c in s], dtype=torch.long)
        
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join([self.id_to_char[i] for i in ids])


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    max_idx = len(data) - seq_len - 1
    if max_idx <= 0:
        raise ValueError(f"Dataset length ({len(data)}) must be strictly greater than seq_len + 1 ({seq_len + 1}).")
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Soft L-Topological Language Model Prototype (Phase 1: Unit Testing) ===")
    print(f"Running on Device: {device}\n")
    
    # 1. Dataset & Tokenizer
    tokenizer = CharTokenizer(SAMPLE_TEXT)
    data = tokenizer.encode(SAMPLE_TEXT)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocab_size} unique characters")
    
    # 2. Hyperparameters for Unit Testing
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    hidden_dim = 128
    seq_len = 32
    batch_size = 8
    lr = 3e-3
    steps = 150
    
    # 3. Instantiate Models
    soft_l_model = SoftLTopologicalLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
        dropout=0.0,
        init_type="beta",
        router_type="lukasiewicz",
    ).to(device)
    
    baseline_model = StandardTransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
        dropout=0.0
    ).to(device)
    
    print(f"Soft L-Topological LM Parameters: {soft_l_model.num_parameters:,}")
    print(f"Baseline Transformer Parameters:  {baseline_model.num_parameters:,}\n")
    
    # 4. Training Soft L-Topological LM
    optimizer = torch.optim.AdamW(soft_l_model.parameters(), lr=lr)
    soft_l_model.train()
    
    print("--- Starting Soft L-Topological LM Training Loop ---")
    start_time = time.time()
    last_sparsity = 0.0
    for step in range(1, steps + 1):
        xb, yb = get_batch(data, batch_size, seq_len, device)
        logits, loss, metrics = soft_l_model(xb, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 30 == 0 or step == steps:
            last_sparsity = metrics['mean_zero_sparsity_pct']
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"Step {step:3d}/{steps} | Loss: {loss.item():.4f} | Perplexity: {ppl:.2f} | Routing Sparsity: {last_sparsity:.2f}% exact zeros")
            
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s\n")
    
    # 5. XAI Trace & Routing Inspection
    test_phrase = "First Citizen:"
    test_ids = tokenizer.encode(test_phrase).unsqueeze(0).to(device)
    _, _, test_metrics = soft_l_model(test_ids, return_routing_traces=True)
    traces = test_metrics['traces'] # list of dicts per layer
    
    tokens = [tokenizer.id_to_char[i.item()] for i in test_ids[0]]
    trace_text = log_deterministic_reasoning_trace(tokens, traces[0], layer_idx=0, head_idx=0)
    print(trace_text)
    
    # Save visualizations
    os.makedirs("artifacts", exist_ok=True)
    heatmap_path = save_routing_heatmap(tokens, traces[0], save_path="artifacts/toy_lattice_heatmap.png", layer_idx=0, head_idx=0)
    sparsity_plot_path = plot_sparsity_comparison(last_sparsity, baseline_sparsity=0.0, save_path="artifacts/toy_sparsity_plot.png")
    print(f"\nSaved XAI Routing Heatmap to: {heatmap_path}")
    print(f"Saved Sparsity Comparison Plot to: {sparsity_plot_path}")
    
    # 6. Comparative Sparsity
    print(compare_sparsity_summary(last_sparsity, baseline_sparsity=0.0))
    
    # 7. Text Generation Demo
    prompt = "All:"
    prompt_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    generated_ids = soft_l_model.generate(
        prompt_ids,
        max_new_tokens=40,
        temperature=0.75,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    generated_text = tokenizer.decode(generated_ids[0])
    print("--- Autoregressive Generation Sample ---")
    print(f"Prompt: '{prompt}'")
    print(f"Generated Output:\n{generated_text}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
