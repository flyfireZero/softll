"""
Soft L-Topological Language Model (soft_ltopology_lm)
Replaces statistical Euclidean vectors and geometric proximity with strict,
continuous mathematical logic and differentiable lattice operators (t-norms, t-conorms)
to reduce language model hallucinations.
"""

from .embedding import (
    SoftSetEmbedding,
    SoftPositionalEncoding,
    LatticeEmbedding,
)
from .lattice_ops import (
    lukasiewicz_tnorm,
    lukasiewicz_tconorm,
    godel_tnorm,
    godel_tconorm,
    product_tnorm,
    probabilistic_sum_tconorm,
    parametric_convex_tnorm,
    parametric_convex_tconorm,
    frank_tnorm,
    frank_tconorm,
    hamacher_tnorm,
    hamacher_tconorm,
    compute_pairwise_lukasiewicz_routing,
    compute_pairwise_parametric_routing,
    compute_pairwise_product_routing,
    compute_pairwise_godel_routing,
    compute_pairwise_frank_routing,
    compute_exact_zero_sparsity,
    compute_effective_sparsity,
)
from .lattice_router import LatticeSequenceRouter
from .model import (
    SoftLTopologicalLM,
    StandardTransformerLM,
    LatticeBlock,
    LatticeFeedForward,
    get_wikitext_suppression_tokens,
)
from .trainer import (
    ModelTrainer,
    TrainingConfig,
    WikiTextDataset,
    get_wikitext_dataloader,
)
from .evaluator import ModelEvaluator
from .xai import (
    log_deterministic_reasoning_trace,
    save_routing_heatmap,
    save_multihead_comparison_heatmap,
    plot_sparsity_comparison,
    compare_sparsity_summary,
    format_autoregressive_trace,
    save_autoregressive_step_heatmap,
)
from .hallucination import HallucinationBenchmark
from .encoder import (
    LatticeEncoderBlock,
    SoftLTopologicalEncoder,
    StandardTransformerEncoder,
)

__all__ = [
    "SoftSetEmbedding",
    "SoftPositionalEncoding",
    "LatticeEmbedding",
    "lukasiewicz_tnorm",
    "lukasiewicz_tconorm",
    "godel_tnorm",
    "godel_tconorm",
    "product_tnorm",
    "probabilistic_sum_tconorm",
    "parametric_convex_tnorm",
    "parametric_convex_tconorm",
    "frank_tnorm",
    "frank_tconorm",
    "hamacher_tnorm",
    "hamacher_tconorm",
    "compute_pairwise_lukasiewicz_routing",
    "compute_pairwise_parametric_routing",
    "compute_pairwise_product_routing",
    "compute_pairwise_godel_routing",
    "compute_pairwise_frank_routing",
    "compute_exact_zero_sparsity",
    "compute_effective_sparsity",
    "LatticeSequenceRouter",
    "SoftLTopologicalLM",
    "StandardTransformerLM",
    "LatticeBlock",
    "LatticeFeedForward",
    "get_wikitext_suppression_tokens",
    "ModelTrainer",
    "TrainingConfig",
    "WikiTextDataset",
    "get_wikitext_dataloader",
    "ModelEvaluator",
    "log_deterministic_reasoning_trace",
    "save_routing_heatmap",
    "save_multihead_comparison_heatmap",
    "plot_sparsity_comparison",
    "compare_sparsity_summary",
    "format_autoregressive_trace",
    "save_autoregressive_step_heatmap",
    "HallucinationBenchmark",
    "LatticeEncoderBlock",
    "SoftLTopologicalEncoder",
    "StandardTransformerEncoder",
]


def main() -> None:
    print("Soft L-Topological Language Model (Final Capstone Review Suite)")
    print("Run `uv run experiments/exp1_convergence.py` or `uv run test.py`.")
