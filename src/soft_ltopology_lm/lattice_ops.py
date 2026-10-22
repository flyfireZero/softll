"""
Lattice Operations for Soft L-Topological Language Modelling.
Implements differentiable t-norms, t-conorms, parametric Frank/Hamacher families,
and high-performance JIT-scripted kernels for continuous sequence routing.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


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
    Parametric Convex t-norm:
    T_lambda(x, y) = lambda * max(0, x + y - 1) + (1 - lambda) * min(x, y)
    """
    luka = torch.clamp(x + y - 1.0, min=0.0, max=1.0)
    godel = torch.minimum(x, y)
    return lambda_val * luka + (1.0 - lambda_val) * godel


@torch.jit.script
def parametric_convex_tconorm(x: torch.Tensor, y: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    r"""
    Parametric Convex t-conorm:
    S_lambda(x, y) = (1 - lambda) * min(1, x + y) + lambda * max(x, y)
    """
    luka_conorm = torch.clamp(x + y, min=0.0, max=1.0)
    godel_conorm = torch.maximum(x, y)
    return (1.0 - lambda_val) * luka_conorm + lambda_val * godel_conorm


@torch.jit.script
def frank_tnorm(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    r"""
    Frank Parametric t-norm Family:
    T_s^F(x, y) = log_s(1 + (s^x - 1)(s^y - 1) / (s - 1)) for s > 0, s != 1.
    Interpolates smoothly between:
    - Minimum / Godel (s -> 0)
    - Product (s -> 1)
    - Lukasiewicz (s -> inf)
    """
    # Numerically stable formulation
    # Clamp s to [1e-3, 50.0]
    s_safe = torch.clamp(s, min=1.001, max=50.0)
    num = (torch.pow(s_safe, x) - 1.0) * (torch.pow(s_safe, y) - 1.0)
    denom = s_safe - 1.0
    arg = torch.clamp(1.0 + num / denom, min=1e-7)
    return torch.clamp(torch.log(arg) / torch.log(s_safe), min=0.0, max=1.0)


@torch.jit.script
def frank_tconorm(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    r"""
    Frank Parametric t-conorm Family:
    S_s^F(x, y) = 1 - T_s^F(1 - x, 1 - y)
    """
    return 1.0 - frank_tnorm(1.0 - x, 1.0 - y, s)


@torch.jit.script
def hamacher_tnorm(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    r"""
    Hamacher Parametric t-norm Family:
    T_gamma^H(x, y) = (x * y) / (gamma + (1 - gamma) * (x + y - x * y))
    gamma=1 is algebraic product; gamma=2 is Einstein product.
    """
    prod = x * y
    denom = gamma + (1.0 - gamma) * (x + y - prod)
    return torch.clamp(prod / torch.clamp(denom, min=1e-7), min=0.0, max=1.0)


@torch.jit.script
def hamacher_tconorm(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    r"""
    Hamacher Parametric t-conorm Family:
    S_gamma^H(x, y) = (x + y - (2 - gamma) * x * y) / (1 - (1 - gamma) * x * y)
    """
    num = x + y - (2.0 - gamma) * (x * y)
    denom = 1.0 - (1.0 - gamma) * (x * y)
    return torch.clamp(num / torch.clamp(denom, min=1e-7), min=0.0, max=1.0)


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
    Computes pairwise token Meet and Join using parametric convex t-norm/t-conorm.
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
    Computes pairwise token Meet and Join using Product t-norm and Probabilistic sum.
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


@torch.jit.script
def compute_pairwise_godel_routing(
    queries: torch.Tensor, 
    keys: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Computes pairwise token Meet and Join using Gödel (Min/Max) t-norm/conorm.
    """
    q_exp = queries.unsqueeze(3)
    k_exp = keys.unsqueeze(2)
    
    meet = torch.minimum(q_exp, k_exp)
    join = torch.maximum(q_exp, k_exp)
    
    meet_agg = torch.mean(meet, dim=-1)
    join_agg = torch.mean(join, dim=-1)
    sparse_meet = torch.clamp(meet_agg - tau, min=0.0)
    containment = sparse_meet / (join_agg + 1e-8)
    return containment, meet_agg, join_agg, meet


@torch.jit.script
def compute_pairwise_frank_routing(
    queries: torch.Tensor, 
    keys: torch.Tensor,
    s_param: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Computes pairwise token Meet and Join using Frank parametric t-norm.
    """
    q_exp = queries.unsqueeze(3)
    k_exp = keys.unsqueeze(2)
    s = s_param.view(1, queries.size(1), 1, 1, 1)
    
    meet = frank_tnorm(q_exp, k_exp, s)
    join = frank_tconorm(q_exp, k_exp, s)
    
    meet_agg = torch.mean(meet, dim=-1)
    join_agg = torch.mean(join, dim=-1)
    sparse_meet = torch.clamp(meet_agg - tau, min=0.0)
    containment = sparse_meet / (join_agg + 1e-8)
    return containment, meet_agg, join_agg, meet


# ============================================================================
# Sparsity & Lattice Metrics
# ============================================================================
def compute_exact_zero_sparsity(tensor: torch.Tensor, atol: float = 1e-6) -> float:
    r"""
    Measures the exact percentage of elements strictly equal to 0.0 (or within atol).
    Unlike standard softmax attention (where weights are strictly > 0),
    the nilpotent t-norms yield exact 0.0 values.
    """
    if tensor.numel() == 0:
        return 0.0
    zero_elements = (tensor <= atol).sum().item()
    total_elements = tensor.numel()
    return float(zero_elements / total_elements) * 100.0


def compute_effective_sparsity(tensor: torch.Tensor, threshold: float = 1e-3) -> float:
    r"""
    Measures effective sparsity: weights <= threshold.
    """
    if tensor.numel() == 0:
        return 0.0
    sparse_elements = (tensor <= threshold).sum().item()
    total_elements = tensor.numel()
    return float(sparse_elements / total_elements) * 100.0
