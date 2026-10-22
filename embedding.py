"""
Soft Set Embedding Layer for Soft L-Topological Language Modelling.
Maps discrete token IDs to continuous lattice features [0, 1]^D using
unbounded logit matrices squashed via Sigmoid activation.
Supports Logit-Sigmoid and Beta distribution initializations.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SoftSetEmbedding(nn.Module):
    r"""
    Soft Set Embedding Layer:
    Replaces Euclidean vector embeddings with parameterized soft sets over [0, 1]^D.
    
    The internal representation uses unbounded real-valued logits squashed through
    a Sigmoid activation, guaranteeing that all feature values strictly reside
    in the continuous lattice L = [0, 1] while remaining fully differentiable.
    
    Initialization Modes:
    - 'logit_sigmoid': Normal distribution in logit space centered negatively
      (e.g., mean=-1.5) so initial sigmoid outputs are centered near 0.18,
      readily satisfying the Lukasiewicz sparsity condition (x + y <= 1.0).
    - 'beta': Samples soft set memberships from a Beta(alpha, beta) distribution
      and maps them into logit space using the inverse sigmoid (logit) function.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_type: str = "beta",
        beta_alpha: float = 1.0,
        beta_beta: float = 3.0,
        logit_mean: float = -1.5,
        init_scale: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_type = init_type.lower()
        
        # Unbounded logit embedding table
        self.logit_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize according to specified technique
        if self.init_type == "beta":
            # Beta distribution initialization: sample u ~ Beta(alpha, beta) in (0, 1)
            # and project via logit(u) = ln(u / (1 - u))
            beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
            u = beta_dist.sample((vocab_size, embed_dim))
            u = torch.clamp(u, min=1e-5, max=1.0 - 1e-5)
            logits = torch.log(u / (1.0 - u))
            with torch.no_grad():
                self.logit_embedding.weight.copy_(logits)
        elif self.init_type == "logit_sigmoid":
            # Direct normal initialization in logit space
            nn.init.normal_(self.logit_embedding.weight, mean=logit_mean, std=init_scale)
        else:
            raise ValueError(f"Unknown init_type '{init_type}'. Expected 'beta' or 'logit_sigmoid'.")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input_ids: [Batch, SeqLen] LongTensor
        Returns:
            soft_sets: [Batch, SeqLen, EmbedDim] FloatTensor strictly in [0, 1]
        """
        logits = self.logit_embedding(input_ids)
        soft_sets = torch.sigmoid(logits)
        return soft_sets


class SoftPositionalEncoding(nn.Module):
    r"""
    Continuous Lattice Positional Encoding:
    Maps sequence indices to bounded soft sets in [0, 1]^D.
    """
    def __init__(self, max_seq_len: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.learnable = learnable
        
        if learnable:
            # Shift positional logits negatively to preserve baseline sparsity
            self.pos_logits = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.1 - 1.2)
        else:
            pe = torch.zeros(max_seq_len, embed_dim)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # Affine squash to [0, 1]
            pe = (pe + 1.0) / 2.0
            self.register_buffer('pe', pe.unsqueeze(0))
            
    def forward(self, seq_len: int) -> torch.Tensor:
        r"""
        Args:
            seq_len: sequence length
        Returns:
            pos_soft_sets: [1, SeqLen, EmbedDim] FloatTensor strictly in [0, 1]
        """
        if self.learnable:
            return torch.sigmoid(self.pos_logits[:, :seq_len, :])
        else:
            return self.pe[:, :seq_len, :]


class LatticeEmbedding(nn.Module):
    r"""
    Full L-Topological Embedding Layer:
    Combines SoftSetEmbedding and SoftPositionalEncoding using continuous lattice operations,
    strictly preserving the [0, 1]^D continuous lattice boundary.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        init_type: str = "beta",
        beta_alpha: float = 1.0,
        beta_beta: float = 3.0,
        learnable_pos: bool = True,
        blend_mode: str = "convex",
    ):
        super().__init__()
        self.tok_emb = SoftSetEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            init_type=init_type,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
        self.pos_emb = SoftPositionalEncoding(
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            learnable=learnable_pos,
        )
        self.blend_mode = blend_mode
        # Learnable convex mixing parameter in [0, 1]
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x_tok = self.tok_emb(input_ids)
        x_pos = self.pos_emb(seq_len)
        
        if self.blend_mode == "convex":
            alpha = torch.sigmoid(self.alpha_logit)
            return alpha * x_tok + (1.0 - alpha) * x_pos
        elif self.blend_mode == "lukasiewicz_conorm":
            return torch.clamp(x_tok + x_pos, max=1.0)
        elif self.blend_mode == "probabilistic_sum":
            return x_tok + x_pos - (x_tok * x_pos)
        else:
            return 0.5 * (x_tok + x_pos)
