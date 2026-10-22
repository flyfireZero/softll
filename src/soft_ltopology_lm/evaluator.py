"""
Comprehensive Evaluator for Soft L-Topological and Baseline Models.
Evaluates:
- Validation Cross-Entropy Loss and Perplexity (PPL)
- Deterministic Routing Sparsity (% exact 0.0 dead connections)
- Multi-Head and Hierarchical Layer Sparsity Distributions
- Inference Latency (ms/token) and Generation Throughput (tokens/sec)
- Memory / VRAM Footprint
"""

import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    @torch.no_grad()
    def evaluate_loss_and_perplexity(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        max_eval_batches: Optional[int] = 50,
    ) -> Dict[str, float]:
        r"""
        Computes mean validation loss, perplexity, and routing sparsity across validation batches.
        """
        model.eval()
        total_loss = 0.0
        total_batches = 0
        total_sparsity = 0.0
        
        for idx, (xb, yb) in enumerate(dataloader):
            if max_eval_batches is not None and idx >= max_eval_batches:
                break
                
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            
            # Check if model returns metrics dict (SoftLTopologicalLM) or just loss (StandardTransformerLM)
            out = model(xb, yb)
            if len(out) == 3:
                logits, loss, metrics = out
                sparsity = metrics.get("mean_zero_sparsity_pct", 0.0)
            else:
                logits, loss = out
                sparsity = 0.0
                
            total_loss += loss.item()
            total_sparsity += sparsity
            total_batches += 1
            
        mean_loss = total_loss / max(total_batches, 1)
        mean_ppl = math.exp(min(mean_loss, 20.0))
        mean_sparsity = total_sparsity / max(total_batches, 1)
        
        return {
            "val_loss": mean_loss,
            "perplexity": mean_ppl,
            "mean_zero_sparsity_pct": mean_sparsity,
            "eval_batches": total_batches,
        }
        
    @torch.no_grad()
    def benchmark_inference_speed(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
    ) -> Dict[str, float]:
        r"""
        Benchmarks autoregressive generation speed (latency and throughput).
        """
        model.eval()
        prompt_ids = prompt_ids.to(self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = model.generate(prompt_ids, max_new_tokens=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
        latencies = []
        for _ in range(benchmark_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(prompt_ids, max_new_tokens=max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            
        avg_time = sum(latencies) / len(latencies)
        tokens_per_sec = max_new_tokens / avg_time
        ms_per_token = (avg_time / max_new_tokens) * 1000.0
        
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        return {
            "avg_latency_sec": avg_time,
            "tokens_per_second": tokens_per_sec,
            "ms_per_token": ms_per_token,
            "max_vram_mb": vram_mb,
        }
