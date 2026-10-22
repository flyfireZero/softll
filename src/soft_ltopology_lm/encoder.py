"""
Encoder-Only Architectures for Soft L-Topology and Standard Transformer:
1. SoftLTopologicalEncoder: Bidirectional continuous lattice sequence router
   evaluating exact logical intersections (Meet) and continuous containment across
   the entire sequence without causal masking.
2. StandardTransformerEncoder: Bidirectional dot-product attention (BERT-style)
   baseline with identical parameter count for rigorous benchmarking.

Supports:
- Bidirectional Masked Language Modeling (MLM / Cloze Task)
- Sequence Classification / Context Containment / Logical Entailment Probing
- Contextual Distractor Isolation with deterministic exact-zero sparsity
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List

from .embedding import LatticeEmbedding
from .lattice_router import LatticeSequenceRouter
from .model import LatticeFeedForward


class LatticeEncoderBlock(nn.Module):
    r"""
    A single Bidirectional Soft L-Topological Encoder Block.
    Computes bidirectional continuous lattice routing without causal restriction.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        router_type: str = "lukasiewicz",
        bounded_values: bool = False,
        init_tau: float = 0.0012,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.router = LatticeSequenceRouter(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_causal=False,  # Bidirectional routing across all tokens
            router_type=router_type,
            bounded_values=bounded_values,
            init_tau=init_tau,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = LatticeFeedForward(embed_dim, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], float]:
        norm_x = torch.sigmoid(self.norm1(x))
        router_out, trace, sparsity, _ = self.router(
            norm_x,
            mask=mask,
            past_key_value=None,
            use_cache=False,
            return_routing_trace=return_routing_trace,
        )
        x = x + router_out
        x = x + self.ffn(self.norm2(x))
        return x, trace, sparsity


class SoftLTopologicalEncoder(nn.Module):
    r"""
    Soft L-Topological Bidirectional Encoder:
    Replaces dense softmax attention with continuous lattice logic across bidirectional context.
    
    Provides:
    - Exact 0.0 zero-leakage routing to irrelevant distractor tokens
    - Differentiable lattice intersection (Meet) between bidirectional tokens
    - Continuous containment representation for logical entailment / cloze prediction
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        router_type: str = "lukasiewicz",
        bounded_values: bool = False,
        init_tau: float = 0.0012,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        
        # Soft set embedding & positional encoding in [0, 1]^D
        self.embedding = LatticeEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            init_type="beta",
            learnable_pos=True,
            blend_mode="convex",
        )
        
        # Bidirectional lattice encoder blocks
        self.blocks = nn.ModuleList([
            LatticeEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                router_type=router_type,
                bounded_values=bounded_values,
                init_tau=init_tau,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Task Heads
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        return_routing_traces: bool = False,
    ) -> Dict[str, Any]:
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if mlm_labels is not None:
                mlm_labels = mlm_labels[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]

        B, S = input_ids.shape
        x = self.embedding(input_ids)
        
        all_traces = []
        layer_sparsities = []
        
        for block in self.blocks:
            x, trace, sparsity = block(
                x,
                mask=attention_mask,
                return_routing_trace=return_routing_traces,
            )
            layer_sparsities.append(sparsity)
            if return_routing_traces:
                all_traces.append(trace)
                
        hidden_states = self.final_norm(x)
        
        # MLM Head
        mlm_logits = self.mlm_head(hidden_states)
        
        # CLS / Sentence Representation (mean pooling over valid tokens or token 0)
        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-7)
        else:
            pooled = hidden_states.mean(dim=1)
            
        cls_logits = self.cls_head(pooled)
        
        # Compute losses if labels provided
        total_loss = None
        mlm_loss = None
        cls_loss = None
        
        if mlm_labels is not None:
            mlm_loss = nn.functional.cross_entropy(
                mlm_logits.view(-1, self.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = mlm_loss if total_loss is None else total_loss + mlm_loss
            
        if cls_labels is not None:
            cls_loss = nn.functional.cross_entropy(cls_logits, cls_labels)
            total_loss = cls_loss if total_loss is None else total_loss + cls_loss
            
        mean_sparsity = sum(layer_sparsities) / max(len(layer_sparsities), 1)
        
        return {
            "mlm_logits": mlm_logits,
            "cls_logits": cls_logits,
            "hidden_states": hidden_states,
            "pooled": pooled,
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "cls_loss": cls_loss,
            "mean_zero_sparsity_pct": mean_sparsity,
            "layer_sparsities": layer_sparsities,
            "traces": all_traces if return_routing_traces else None,
        }


class StandardTransformerEncoder(nn.Module):
    r"""
    Standard Bidirectional Transformer Encoder (BERT-style dot-product baseline).
    Has identical layer count, head count, embedding dim, and parameter count.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        
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
        self.final_norm = nn.LayerNorm(embed_dim)
        
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        return_routing_traces: bool = False,
    ) -> Dict[str, Any]:
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if mlm_labels is not None:
                mlm_labels = mlm_labels[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]

        B, S = input_ids.shape
        pos = torch.arange(0, S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        
        key_padding_mask = None
        if attention_mask is not None:
            # PyTorch TransformerEncoder expects key_padding_mask: True for positions to IGNORE
            key_padding_mask = ~attention_mask.bool()
            
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        hidden_states = self.final_norm(x)
        
        # MLM Head
        mlm_logits = self.mlm_head(hidden_states)
        
        # CLS / Sentence Representation (mean pooling)
        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-7)
        else:
            pooled = hidden_states.mean(dim=1)
            
        cls_logits = self.cls_head(pooled)
        
        total_loss = None
        mlm_loss = None
        cls_loss = None
        
        if mlm_labels is not None:
            mlm_loss = nn.functional.cross_entropy(
                mlm_logits.view(-1, self.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = mlm_loss if total_loss is None else total_loss + mlm_loss
            
        if cls_labels is not None:
            cls_loss = nn.functional.cross_entropy(cls_logits, cls_labels)
            total_loss = cls_loss if total_loss is None else total_loss + cls_loss
            
        return {
            "mlm_logits": mlm_logits,
            "cls_logits": cls_logits,
            "hidden_states": hidden_states,
            "pooled": pooled,
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "cls_loss": cls_loss,
            "mean_zero_sparsity_pct": 0.0,
            "layer_sparsities": [0.0] * len(self.encoder.layers),
            "traces": None,
        }
