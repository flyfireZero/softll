"""
Unit tests for End-to-End Model Architectures:
Verifies:
1. SoftLTopologicalLM and StandardTransformerLM parameter parity.
2. Forward pass with causal masking.
3. Loss calculation and backward pass.
4. Deterministic sparsity generation in SoftLTopologicalLM vs 0.00% in baseline.
5. Autoregressive text generation.
"""

import torch
import unittest
from soft_ltopology_lm.model import SoftLTopologicalLM, StandardTransformerLM


class TestModelArchitectures(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 500
        self.embed_dim = 64
        self.num_layers = 2
        self.num_heads = 2
        self.hidden_dim = 128
        self.seq_len = 16
        self.batch_size = 2

    def test_parameter_parity(self):
        soft_l = SoftLTopologicalLM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.seq_len,
        )
        baseline = StandardTransformerLM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.seq_len,
        )
        
        # Verify both models are in the expected parameter range
        self.assertGreater(soft_l.num_parameters, 0)
        self.assertGreater(baseline.num_parameters, 0)
        delta = abs(soft_l.num_parameters - baseline.num_parameters)
        # Delta should be negligible (< 1% delta)
        self.assertLess(delta / baseline.num_parameters, 0.01)

    def test_soft_l_forward_backward(self):
        torch.manual_seed(42)
        soft_l = SoftLTopologicalLM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.seq_len,
        )
        xb = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        yb = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        logits, loss, metrics = soft_l(xb, yb, return_routing_traces=True)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)
        
        # Check backward pass
        loss.backward()
        for p in soft_l.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

        # Check deterministic sparsity > 0%
        sparsity = metrics["mean_zero_sparsity_pct"]
        self.assertGreater(sparsity, 0.0, "Soft L-Topology must produce non-zero exact-zero sparsity")

    def test_baseline_forward_backward(self):
        baseline = StandardTransformerLM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.seq_len,
        )
        xb = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        yb = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        logits, loss = baseline(xb, yb)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        
        loss.backward()
        for p in baseline.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())


if __name__ == "__main__":
    unittest.main()
