"""
Complete Model Architectures:
1. SoftLTopologicalLM: Soft L-Topological Language Model utilizing continuous lattice sequence routing.
2. StandardTransformerLM: Standard Dot-Product Multi-Head Attention Baseline for exact 1:1 parameter comparison.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List

from embedding import LatticeEmbedding, SoftSetEmbedding, SoftPositionalEncoding
from lattice_router import LatticeSequenceRouter


class LatticeFeedForward(nn.Module):
    r"""
    Feed-Forward Network adapted for Soft L-Topology.
    Uses GELU gates to project features.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class LatticeBlock(nn.Module):
    r"""
    A single Soft L-Topological Transformer Block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        router_type: str = "lukasiewicz",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.router = LatticeSequenceRouter(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_causal=True,
            router_type=router_type,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = LatticeFeedForward(embed_dim, hidden_dim, dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], float]:
        # LayerNorm mapped into [0, 1] soft sets for router
        norm_x = torch.sigmoid(self.norm1(x))
        router_out, trace, sparsity = self.router(
            norm_x, mask=mask, return_routing_trace=return_routing_trace
        )
        x = x + router_out
        x = x + self.ffn(self.norm2(x))
        return x, trace, sparsity


def apply_repetition_penalty(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    penalty: float = 1.2
) -> torch.Tensor:
    r"""
    Applies frequency/repetition penalty to logits for tokens that have already appeared
    in the context, discouraging degenerate loops while keeping logical coherence.
    """
    if penalty != 1.0 and context_tokens.numel() > 0:
        unique_tokens = torch.unique(context_tokens)
        gathered = logits.gather(1, unique_tokens.unsqueeze(0))
        penalized = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
        logits = logits.scatter(1, unique_tokens.unsqueeze(0), penalized)
    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    r"""
    Filters logits using Top-K and Nucleus (Top-P) probability mass filtering.
    """
    if top_k is not None and top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def get_wikitext_suppression_tokens(tokenizer) -> List[int]:
    r"""
    Identifies all token IDs containing WikiText markup characters ('<', '>', '@', '=')
    or 'unk' to prevent generation from degenerating into raw dataset tags (<unk>, @-@, = = =).
    """
    bad_chars = {'<', '>', '@', '='}
    bad_tokens = set()
    for token, idx in tokenizer.get_vocab().items():
        if any(c in token for c in bad_chars) or token.strip().lower() in {'unk', '<unk>'}:
            bad_tokens.add(idx)
    return sorted(list(bad_tokens))


class SoftLTopologicalLM(nn.Module):
    r"""
    Soft L-Topological Language Model:
    Complete autoregressive sub-word / character LM driven by continuous lattice operators.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        init_type: str = "beta",
        beta_alpha: float = 1.0,
        beta_beta: float = 3.0,
        router_type: str = "lukasiewicz",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.router_type = router_type
        
        # Soft set embedding & positional encoding in [0, 1]^D
        self.embedding = LatticeEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            init_type=init_type,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            learnable_pos=True,
            blend_mode="convex",
        )
        
        # Stack of Lattice Blocks
        self.blocks = nn.ModuleList([
            LatticeBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                router_type=router_type,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head if matching dimensions (optional standard practice)
        # Here kept separate to maintain exact parameter match with baseline
        
        # Parameter count
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_routing_traces: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        r"""
        Args:
            input_ids: [Batch, SeqLen]
            targets: Optional [Batch, SeqLen] for Cross-Entropy Loss
            return_routing_traces: If True, logs attention/routing matrices for XAI.
        """
        # Ensure sequence length does not exceed max_seq_len
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            if targets is not None:
                targets = targets[:, -self.max_seq_len:]
                
        # Embed strictly into [0, 1]^D continuous lattice
        x = self.embedding(input_ids)
        
        all_traces = []
        layer_sparsities = []
        
        for block in self.blocks:
            x, trace, sparsity = block(x, return_routing_trace=return_routing_traces)
            layer_sparsities.append(sparsity)
            if return_routing_traces:
                all_traces.append(trace)
                
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
            
        metrics = {
            "mean_zero_sparsity_pct": sum(layer_sparsities) / max(len(layer_sparsities), 1),
            "layer_sparsities": layer_sparsities,
            "traces": all_traces if return_routing_traces else None,
        }
        return logits, loss, metrics

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        suppress_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Enhanced autoregressive generation with top-k, top-p, and repetition penalty."""
        self.eval()
        curr_ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = curr_ids[:, -self.max_seq_len:]
            logits, _, _ = self(idx_cond)
            next_logits = logits[:, -1, :].clone()
            
            # Apply repetition penalty to recent context
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, curr_ids[0], penalty=repetition_penalty)
                
            # Suppress unwanted tokens (e.g. unk)
            if suppress_token_ids:
                for bad_id in suppress_token_ids:
                    next_logits[:, bad_id] = -float("Inf")
                    
            next_logits = next_logits / max(temperature, 1e-5)
            filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return curr_ids

    @torch.no_grad()
    def generate_with_explanation(
        self,
        prompt_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 15,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        suppress_token_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        r"""
        Generates text step-by-step while logging exact lattice routing context and candidates,
        providing total mathematical explainability of every generated token.
        """
        self.eval()
        curr_ids = prompt_ids.clone()
        step_logs = []
        
        for step in range(1, max_new_tokens + 1):
            idx_cond = curr_ids[:, -self.max_seq_len:]
            logits, _, metrics = self(idx_cond, return_routing_traces=True)
            next_logits = logits[:, -1, :].clone()
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, curr_ids[0], penalty=repetition_penalty)
                
            if suppress_token_ids:
                for bad_id in suppress_token_ids:
                    next_logits[:, bad_id] = -float("Inf")
                    
            next_logits = next_logits / max(temperature, 1e-5)
            filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            chosen_id = next_token.item()
            chosen_str = tokenizer.decode([chosen_id])
            chosen_prob = probs[0, chosen_id].item()
            
            # Top candidate alternatives
            top_k_show = min(4, probs.size(-1))
            top_vals, top_indices = torch.topk(probs[0], top_k_show)
            candidates = [(tokenizer.decode([idx.item()]), val.item()) for val, idx in zip(top_vals, top_indices)]
            
            # Extract routing weights for this step from top layer (Layer 3)
            last_trace = metrics['traces'][-1]
            norm_routing = last_trace["norm_routing"][0]  # [Heads, S, S]
            query_routing = norm_routing[:, -1, :]        # [Heads, S]
            
            num_heads, S = query_routing.shape
            cond_tokens = [tokenizer.decode([t.item()]) for t in idx_cond[0]]
            
            # 1. Exact-zero path sparsity for this token query across all attention heads
            dead_paths = (query_routing <= 1e-4).sum().item()
            total_paths = query_routing.numel()
            query_sparsity_pct = (dead_paths / total_paths) * 100.0
            
            # 2. Per-token aggregated routing (mean and max across heads)
            mean_weights = query_routing.mean(dim=0)
            max_weights, best_heads = query_routing.max(dim=0)
            
            # Active antecedents: tokens with non-zero containment in at least one head
            active_sources = []
            pruned_sources = []
            for j in range(S):
                tok_str = cond_tokens[j]
                max_w = max_weights[j].item()
                mean_w = mean_weights[j].item()
                head_id = best_heads[j].item()
                active_h_count = (query_routing[:, j] > 1e-4).sum().item()
                if max_w > 1e-4:
                    active_sources.append({
                        "token": tok_str,
                        "mean_weight": mean_w,
                        "max_weight": max_w,
                        "primary_head": head_id,
                        "active_heads": active_h_count,
                    })
                else:
                    pruned_sources.append(tok_str)
                    
            # Per-head focus: top antecedent and head-level sparsity
            head_details = []
            for h in range(num_heads):
                h_row = query_routing[h]
                h_zeros = (h_row <= 1e-4).sum().item()
                h_top_idx = h_row.argmax().item()
                head_details.append({
                    "head": h,
                    "zeros": h_zeros,
                    "total": S,
                    "sparsity_pct": (h_zeros / S) * 100.0,
                    "top_token": cond_tokens[h_top_idx],
                    "top_weight": h_row[h_top_idx].item(),
                })
                
            step_logs.append({
                "step": step,
                "chosen_token": chosen_str,
                "chosen_prob": chosen_prob,
                "candidates": candidates,
                "query_sparsity_pct": query_sparsity_pct,
                "dead_paths": dead_paths,
                "total_paths": total_paths,
                "model_sparsity_pct": metrics["mean_zero_sparsity_pct"],
                "active_sources": active_sources,
                "pruned_sources": pruned_sources,
                "head_details": head_details,
                "all_cond_tokens": list(cond_tokens),
                "mean_routing_row": mean_weights.cpu().numpy(),
            })
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        return curr_ids, step_logs


# ============================================================================
# Baseline Dot-Product Transformer (for 1:1 Benchmark Comparison)
# ============================================================================

class StandardTransformerLM(nn.Module):
    r"""
    Standard Dot-Product Attention Transformer for equal-parameter baseline benchmarking.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            if targets is not None:
                targets = targets[:, -self.max_seq_len:]
                
        B, S = input_ids.shape
        pos = torch.arange(0, S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        suppress_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Enhanced autoregressive generation for baseline."""
        self.eval()
        curr_ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = curr_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :].clone()
            
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, curr_ids[0], penalty=repetition_penalty)
                
            if suppress_token_ids:
                for bad_id in suppress_token_ids:
                    next_logits[:, bad_id] = -float("Inf")
                    
            next_logits = next_logits / max(temperature, 1e-5)
            filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return curr_ids
