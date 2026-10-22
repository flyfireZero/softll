"""
Lattice Operations for Soft L-Topological Language Modelling.
Implements differentiable t-norms, t-conorms, parametric convex combinations,
and high-performance JIT-scripted kernels for sequence routing.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ============================================================================
# PyTorch JIT Scripted Core Lattice Operators
# ============================================================================

@torch.jit.script
def lukasiewicz_tnorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Łukasiewicz t-norm (Intersection / Meet):
    T_L(x, y) = max(0, x + y - 1)
    Yields STRICT 0.0 when x + y <= 1.0, enabling true mathematical sparsity.
    """
    return torch.clamp(x + y - 1.0, min=0.0, max=1.0)


@torch.jit.script
def lukasiewicz_tconorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Łukasiewicz t-conorm (Union / Join):
    S_L(x, y) = min(1, x + y)
    """
    return torch.clamp(x + y, min=0.0, max=1.0)


@torch.jit.script
def godel_tnorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Gödel / Minimum t-norm (Intersection / Meet):
    T_G(x, y) = min(x, y)
    """
    return torch.minimum(x, y)


@torch.jit.script
def godel_tconorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Gödel / Maximum t-conorm (Union / Join):
    S_G(x, y) = max(x, y)
    """
    return torch.maximum(x, y)


@torch.jit.script
def product_tnorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Product t-norm (Intersection / Meet):
    T_P(x, y) = x * y
    """
    return x * y


@torch.jit.script
def probabilistic_sum_tconorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Probabilistic Sum t-conorm (Union / Join):
    S_P(x, y) = x + y - x * y
    """
    return x + y - (x * y)


@torch.jit.script
def parametric_convex_tnorm(x: torch.Tensor, y: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    r"""
    Parametric Convex t-norm (Page 9 of Architecture Slide):
    T_lambda(x, y) = lambda * max(0, x + y - 1) + (1 - lambda) * min(x, y)
    """
    luka = torch.clamp(x + y - 1.0, min=0.0, max=1.0)
    godel = torch.minimum(x, y)
    return lambda_val * luka + (1.0 - lambda_val) * godel


@torch.jit.script
def parametric_convex_tconorm(x: torch.Tensor, y: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    r"""
    Parametric Convex t-conorm (Page 9 of Architecture Slide):
    S_lambda(x, y) = (1 - lambda) * min(1, x + y) + lambda * max(x, y)
    """
    luka_conorm = torch.clamp(x + y, min=0.0, max=1.0)
    godel_conorm = torch.maximum(x, y)
    return (1.0 - lambda_val) * luka_conorm + lambda_val * godel_conorm


# ============================================================================
# Pairwise Lattice Sequence Kernels (JIT-scripted)
# ============================================================================

@torch.jit.script
def compute_pairwise_lukasiewicz_routing(
    queries: torch.Tensor, 
    keys: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Computes pairwise token Meet (Intersection) and Join (Union) across sequence dimension.
    
    Args:
        queries: [B, H, S_q, D] tensor in [0, 1]
        keys:    [B, H, S_k, D] tensor in [0, 1]
        tau:     [1, H, 1, 1] threshold tensor
        
    Returns:
        containment:  [B, H, S_q, S_k] continuous lattice Jaccard index
        meet_agg:     [B, H, S_q, S_k] aggregated Meet degree in [0, 1]
        join_agg:     [B, H, S_q, S_k] aggregated Join degree in [0, 1]
        meet_feature: [B, H, S_q, S_k, D] element-wise feature intersections
    """
    q_exp = queries.unsqueeze(3)
    k_exp = keys.unsqueeze(2)
    
    # Feature-wise Łukasiewicz Meet and Join
    meet = torch.clamp(q_exp + k_exp - 1.0, min=0.0, max=1.0)
    join = torch.clamp(q_exp + k_exp, min=0.0, max=1.0)
    
    # Aggregate over latent features D: [B, H, S_q, S_k]
    meet_agg = torch.mean(meet, dim=-1)
    join_agg = torch.mean(join, dim=-1)
    
    # Nilpotent logical thresholding
    sparse_meet = torch.clamp(meet_agg - tau, min=0.0)
    containment = sparse_meet / (join_agg + 1e-8)
    return containment, meet_agg, join_agg, meet


@torch.jit.script
def compute_pairwise_parametric_routing(
    queries: torch.Tensor, 
    keys: torch.Tensor, 
    lambda_param: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Computes pairwise token Meet and Join using parametric convex t-norm/t-conorm (Slide 9).
    """
    q_exp = queries.unsqueeze(3)
    k_exp = keys.unsqueeze(2)
    lam = lambda_param.view(1, queries.size(1), 1, 1, 1)
    
    meet = parametric_convex_tnorm(q_exp, k_exp, lam)
    join = parametric_convex_tconorm(q_exp, k_exp, lam)
    
    meet_agg = torch.mean(meet, dim=-1)
    join_agg = torch.mean(join, dim=-1)
    sparse_meet = torch.clamp(meet_agg - tau, min=0.0)
    containment = sparse_meet / (join_agg + 1e-8)
    return containment, meet_agg, join_agg, meet


@torch.jit.script
def compute_pairwise_product_routing(
    queries: torch.Tensor, 
    keys: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Computes pairwise token Meet and Join using Product t-norm and Probabilistic sum (Slide 10).
    """
    q_exp = queries.unsqueeze(3)
    k_exp = keys.unsqueeze(2)
    
    meet = q_exp * k_exp
    join = q_exp + k_exp - (q_exp * k_exp)
    
    meet_agg = torch.mean(meet, dim=-1)
    join_agg = torch.mean(join, dim=-1)
    sparse_meet = torch.clamp(meet_agg - tau, min=0.0)
    containment = sparse_meet / (join_agg + 1e-8)
    return containment, meet_agg, join_agg, meet


# Legacy compatibility
@torch.jit.script
def compute_pairwise_lattice_intersection(
    queries: torch.Tensor, 
    keys: torch.Tensor, 
    lambda_param: torch.Tensor
) -> torch.Tensor:
    tau = torch.zeros((1, queries.size(1), 1, 1), device=queries.device, dtype=queries.dtype)
    containment, _, _, _ = compute_pairwise_lukasiewicz_routing(queries, keys, tau)
    return containment


# ============================================================================
# Sparsity & Lattice Metrics
# ============================================================================
def compute_exact_zero_sparsity(tensor: torch.Tensor, atol: float = 1e-6) -> float:
    r"""
    Measures the exact percentage of elements strictly equal to 0.0 (or within atol).
    Unlike standard softmax attention (where weights are always > 0),
    the Lukasiewicz t-norm yields exact 0.0 values.
    """
    if tensor.numel() == 0:
        return 0.0
    zero_elements = (tensor <= atol).sum().item()
    total_elements = tensor.numel()
    return float(zero_elements / total_elements) * 100.0
