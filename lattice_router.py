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

from lattice_ops import (
    compute_pairwise_lukasiewicz_routing,
    compute_pairwise_parametric_routing,
    compute_pairwise_product_routing,
    compute_exact_zero_sparsity,
)


class LatticeSequenceRouter(nn.Module):
    r"""
    Continuous Multi-Head Lattice Router:
    Replaces dot-product self-attention with continuous lattice operations.
    
    Computes:
    - Meet / Intersection: T(Q, K) measuring shared semantic logic
    - Join / Union: S(Q, K) measuring combined contextual span
    - Continuous Lattice Jaccard Containment: T(Q, K) / S(Q, K)
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
        init_tau: float = 0.0012,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.router_type = router_type.lower()
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Bias queries and keys so sigmoid output is initially ~0.42
        nn.init.constant_(self.q_proj.bias, -0.35)
        nn.init.constant_(self.k_proj.bias, -0.35)
        
        # Parametric lambda parameter for convex t-norm/conorm (Slide 9)
        self.lambda_logits = nn.Parameter(torch.empty(1, num_heads, 1, 1, 1))
        nn.init.constant_(self.lambda_logits, 2.5)  # sigmoid(2.5) ~ 0.92 heavily favors Lukasiewicz
        
        # Multi-scale learnable logical sparsity threshold across attention heads:
        # Lower thresholds capture broad contextual overlap; higher thresholds enforce strict logical deductions.
        init_taus = torch.linspace(0.0010, 0.0035, steps=num_heads).view(1, num_heads, 1, 1)
        self.tau_param = nn.Parameter(init_taus)
        
        self.dropout = nn.Dropout(dropout)
        
    @property
    def lambda_param(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_logits)
        
    @property
    def tau(self) -> torch.Tensor:
        # Enforce minimum threshold to mathematically preserve deterministic exact-zero sparsity
        return torch.clamp(self.tau_param, min=0.0012, max=0.05)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], float]:
        B, S, D = x.shape
        
        # Project into [0, 1] continuous lattice soft sets
        Q = torch.sigmoid(self.q_proj(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = torch.sigmoid(self.k_proj(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = torch.sigmoid(self.v_proj(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        tau = self.tau
        
        # Compute Meet, Join, and Relative Containment
        if self.router_type == "lukasiewicz":
            containment, meet, join, meet_feature = compute_pairwise_lukasiewicz_routing(Q, K, tau)
        elif self.router_type == "parametric":
            containment, meet, join, meet_feature = compute_pairwise_parametric_routing(Q, K, self.lambda_param, tau)
        elif self.router_type == "product":
            containment, meet, join, meet_feature = compute_pairwise_product_routing(Q, K, tau)
        else:
            containment, meet, join, meet_feature = compute_pairwise_lukasiewicz_routing(Q, K, tau)
            
        routing_weights = containment
        
        # Causal & Attention Masking
        if self.is_causal:
            causal_mask = torch.tril(torch.ones((S, S), device=x.device, dtype=torch.bool))
            causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, S, S)
            
            # Compute sparsity ONLY over valid causal positions (j <= i)
            valid_routing_weights = routing_weights[causal_mask_expanded]
            sparsity_pct = compute_exact_zero_sparsity(valid_routing_weights)
            
            # Zero-out future tokens
            routing_weights = routing_weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, S, S)
                valid_routing_weights = routing_weights[mask_expanded]
                sparsity_pct = compute_exact_zero_sparsity(valid_routing_weights)
                routing_weights = routing_weights.masked_fill(~mask_expanded, 0.0)
            else:
                sparsity_pct = compute_exact_zero_sparsity(routing_weights)
            
        # Lattice Normalization (Slide 9)
        normalizer = routing_weights.sum(dim=-1, keepdim=True)
        # Fallback to self-identity diagonal if a token has no intersecting context
        eye = torch.eye(S, device=x.device, dtype=routing_weights.dtype).unsqueeze(0).unsqueeze(0)
        safe_normalizer = torch.where(normalizer > 1e-7, normalizer, torch.ones_like(normalizer))
        norm_routing = torch.where(normalizer > 1e-7, routing_weights / safe_normalizer, eye)
        
        norm_routing_dropped = self.dropout(norm_routing)
        
        # Aggregate contextual values
        out = torch.matmul(norm_routing_dropped, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
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
            
        return out, trace, sparsity_pct
