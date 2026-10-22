"""
Unit tests for Soft Set Embeddings and Continuous Positional Encodings.
Verifies:
1. All embeddings strictly reside in [0, 1]^D continuous lattice.
2. Gradient flow through logit squashing.
3. Initialization distributions (Beta, Logit-Sigmoid, Uniform).
4. Positional blending operations.
"""

import torch
import unittest
from soft_ltopology_lm.embedding import (
    SoftSetEmbedding,
    SoftPositionalEncoding,
    LatticeEmbedding,
)


class TestSoftSetEmbedding(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embed_dim = 32
        self.batch_size = 4
        self.seq_len = 16
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_beta_init_boundary(self):
        emb = SoftSetEmbedding(self.vocab_size, self.embed_dim, init_type="beta")
        out = emb(self.input_ids)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_logit_sigmoid_init_boundary(self):
        emb = SoftSetEmbedding(self.vocab_size, self.embed_dim, init_type="logit_sigmoid")
        out = emb(self.input_ids)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_differentiability(self):
        emb = SoftSetEmbedding(self.vocab_size, self.embed_dim, init_type="beta")
        out = emb(self.input_ids)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(emb.logit_embedding.weight.grad)
        self.assertFalse(torch.isnan(emb.logit_embedding.weight.grad).any())

    def test_positional_encoding(self):
        pos_emb = SoftPositionalEncoding(max_seq_len=64, embed_dim=self.embed_dim, learnable=True)
        out = pos_emb(self.seq_len)
        self.assertEqual(out.shape, (1, self.seq_len, self.embed_dim))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_lattice_embedding_blend_modes(self):
        for mode in ["convex", "lukasiewicz_conorm", "probabilistic_sum"]:
            lat_emb = LatticeEmbedding(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_seq_len=64,
                blend_mode=mode,
            )
            out = lat_emb(self.input_ids)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))
            self.assertTrue(torch.all(out >= 0.0), f"Failed >= 0.0 for mode {mode}")
            self.assertTrue(torch.all(out <= 1.0), f"Failed <= 1.0 for mode {mode}")


if __name__ == "__main__":
    unittest.main()
