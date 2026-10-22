"""
Lattice Sequence Router for Soft L-Topological Language Modelling.
Replaces standard Multi-Head Attention (dot-product + softmax) with
Continuous Lattice Routing using differentiable t-norms and t-conorms.
Implements exact logical intersections (Meet), unions (Join), and
continuous lattice Jaccard containment with deterministic sparsity.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union

from .lattice_ops import (
    compute_pairwise_lukasiewicz_routing,
    compute_pairwise_parametric_routing,
    compute_pairwise_product_routing,
    compute_pairwise_godel_routing,
    compute_pairwise_frank_routing,
    compute_exact_zero_sparsity,
    compute_effective_sparsity,
)


class LatticeSequenceRouter(nn.Module):
    r"""
    Continuous Multi-Head Lattice Router:
    Replaces dot-product self-attention with continuous lattice operations.
    
    Computes:
    - Meet / Intersection: T(Q, K) measuring shared semantic logic
    - Join / Union: S(Q, K) measuring combined contextual span
    - Continuous Lattice Jaccard Containment: T(Q, K) / S(Q, K)
    - Nilpotent thresholding for deterministic exact-zero sparsity
    - Row-wise Lattice Normalization
    
    Guarantees exact 0.0 routing sparsity for tokens with no overlapping context.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        router_type: str = "lukasiewicz",
        bounded_values: bool = True,
        init_tau: float = 0.0012,
        min_tau: float = 0.0012,
        max_tau: float = 0.05,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.router_type = router_type.lower()
        self.bounded_values = bounded_values
        self.min_tau = min_tau
        self.max_tau = max_tau
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Bias queries and keys so sigmoid output is initially ~0.42
        nn.init.constant_(self.q_proj.bias, -0.35)
        nn.init.constant_(self.k_proj.bias, -0.35)
        
        # Parametric lambda parameter for convex t-norm/conorm
        self.lambda_logits = nn.Parameter(torch.empty(1, num_heads, 1, 1, 1))
        nn.init.constant_(self.lambda_logits, 2.5)  # sigmoid(2.5) ~ 0.92 heavily favors Lukasiewicz
        
        # Frank parameter s
        self.frank_s_param = nn.Parameter(torch.full((1, num_heads, 1, 1, 1), 10.0))
        
        # Multi-scale learnable logical sparsity threshold across attention heads:
        # Lower thresholds capture broad contextual overlap; higher thresholds enforce strict logical deductions.
        init_taus = torch.linspace(init_tau, init_tau * 3.0, steps=num_heads).view(1, num_heads, 1, 1)
        self.tau_param = nn.Parameter(init_taus)
        
        self.dropout = nn.Dropout(dropout)
        
    @property
    def lambda_param(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_logits)
        
    @property
    def tau(self) -> torch.Tensor:
        # Clamp threshold to mathematically preserve deterministic exact-zero sparsity
        return torch.clamp(self.tau_param, min=self.min_tau, max=self.max_tau)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_routing_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], float, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, D = x.shape
        
        # Project Q and K into [0, 1] continuous lattice soft sets
        Q = torch.sigmoid(self.q_proj(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = torch.sigmoid(self.k_proj(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Value projection: bounded in [0, 1] or unbounded in R^D
        if self.bounded_values:
            V = torch.sigmoid(self.v_proj(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            
        # KV Cache concatenation for fast autoregression
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
            
        present_key_value = (K, V) if use_cache else None
        
        tau = self.tau
        S_q = Q.size(2)
        S_k = K.size(2)
        
        # Compute Meet, Join, and Relative Containment
        if self.router_type == "lukasiewicz":
            containment, meet, join, meet_feature = compute_pairwise_lukasiewicz_routing(Q, K, tau)
        elif self.router_type == "parametric":
            containment, meet, join, meet_feature = compute_pairwise_parametric_routing(Q, K, self.lambda_param, tau)
        elif self.router_type == "product":
            containment, meet, join, meet_feature = compute_pairwise_product_routing(Q, K, tau)
        elif self.router_type == "godel":
            containment, meet, join, meet_feature = compute_pairwise_godel_routing(Q, K, tau)
        elif self.router_type == "frank":
            containment, meet, join, meet_feature = compute_pairwise_frank_routing(Q, K, self.frank_s_param, tau)
        else:
            containment, meet, join, meet_feature = compute_pairwise_lukasiewicz_routing(Q, K, tau)
            
        routing_weights = containment
        
        # Causal & Attention Masking
        if self.is_causal:
            if S_q == 1 and S_k > 1:
                # In KV-cached autoregression at time t, all cached keys are prior tokens (j <= t)
                sparsity_pct = compute_exact_zero_sparsity(routing_weights)
            else:
                causal_mask = torch.tril(torch.ones((S_q, S_k), device=x.device, dtype=torch.bool))
                causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, S_q, S_k)
                valid_routing_weights = routing_weights[causal_mask_expanded]
                sparsity_pct = compute_exact_zero_sparsity(valid_routing_weights)
                routing_weights = routing_weights.masked_fill(~causal_mask_expanded, 0.0)
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, S_q, S_k)
                valid_routing_weights = routing_weights[mask_expanded]
                sparsity_pct = compute_exact_zero_sparsity(valid_routing_weights)
                routing_weights = routing_weights.masked_fill(~mask_expanded, 0.0)
            else:
                sparsity_pct = compute_exact_zero_sparsity(routing_weights)
            
        # Lattice Normalization
        normalizer = routing_weights.sum(dim=-1, keepdim=True)
        safe_normalizer = torch.where(normalizer > 1e-7, normalizer, torch.ones_like(normalizer))
        norm_routing = torch.where(normalizer > 1e-7, routing_weights / safe_normalizer, torch.zeros_like(routing_weights))
        
        # If all past connections evaluate to zero, fall back to self identity
        zero_mask = (normalizer <= 1e-7)
        if zero_mask.any():
            if S_q == S_k:
                eye = torch.eye(S_q, device=x.device, dtype=norm_routing.dtype).unsqueeze(0).unsqueeze(0)
                norm_routing = torch.where(zero_mask, eye, norm_routing)
            else:
                identity_weights = torch.zeros_like(norm_routing)
                identity_weights[..., -1] = 1.0
                norm_routing = torch.where(zero_mask, identity_weights, norm_routing)
        
        norm_routing_dropped = self.dropout(norm_routing)
        
        # Aggregate contextual values
        out = torch.matmul(norm_routing_dropped, V)
        out = out.transpose(1, 2).contiguous().view(B, S_q, D)
        out = self.out_proj(out)
        
        trace = None
        if return_routing_trace:
            trace = {
                "containment": routing_weights,
                "meet": meet,
                "join": join,
                "norm_routing": norm_routing,
                "meet_feature": meet_feature,
            }
            
        return out, trace, sparsity_pct, present_key_value
