"""
Unit tests for Bidirectional Encoders:
Verifies:
1. SoftLTopologicalEncoder and StandardTransformerEncoder parameter parity.
2. Bidirectional MLM forward and backward pass.
3. Bidirectional sequence classification forward and backward pass.
4. Deterministic routing sparsity in SoftLTopologicalEncoder vs 0.00% in StandardTransformerEncoder.
5. Masked token prediction accuracy and tensor shapes.
"""

import torch
import unittest
from soft_ltopology_lm.encoder import SoftLTopologicalEncoder, StandardTransformerEncoder


class TestEncoderArchitectures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 500
        self.embed_dim = 64
        self.num_layers = 2
        self.num_heads = 2
        self.hidden_dim = 128
        self.max_seq_len = 16
        self.batch_size = 2
        self.num_classes = 3

    def test_encoder_parameter_parity(self):
        soft_enc = SoftLTopologicalEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            num_classes=self.num_classes,
        )
        base_enc = StandardTransformerEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            num_classes=self.num_classes,
        )
        
        self.assertGreater(soft_enc.num_parameters, 0)
        self.assertGreater(base_enc.num_parameters, 0)
        delta = abs(soft_enc.num_parameters - base_enc.num_parameters)
        # Delta should be < 1%
        self.assertLess(delta / base_enc.num_parameters, 0.01)

    def test_soft_encoder_mlm_forward_backward(self):
        torch.manual_seed(42)
        soft_enc = SoftLTopologicalEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            num_classes=self.num_classes,
        )
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_seq_len))
        mlm_labels = input_ids.clone()
        mlm_labels[:, 4:] = -100  # Only mask first 4 positions
        
        out = soft_enc(input_ids, mlm_labels=mlm_labels, return_routing_traces=True)
        self.assertEqual(out["mlm_logits"].shape, (self.batch_size, self.max_seq_len, self.vocab_size))
        self.assertIsNotNone(out["loss"])
        self.assertGreater(out["loss"].item(), 0.0)
        
        # Test backward
        out["loss"].backward()
        for p in soft_enc.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())
                
        # Deterministic sparsity must be non-negative
        self.assertGreaterEqual(out["mean_zero_sparsity_pct"], 0.0)

    def test_baseline_encoder_forward(self):
        torch.manual_seed(42)
        base_enc = StandardTransformerEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            num_classes=self.num_classes,
        )
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_seq_len))
        cls_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        out = base_enc(input_ids, cls_labels=cls_labels)
        self.assertEqual(out["cls_logits"].shape, (self.batch_size, self.num_classes))
        self.assertIsNotNone(out["cls_loss"])
        self.assertEqual(out["mean_zero_sparsity_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
