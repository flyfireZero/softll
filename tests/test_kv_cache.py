"""
Unit tests for Continuous Lattice KV Cache and Fast Autoregressive Decoding.
Verifies:
1. KV cache shape and concatenation consistency.
2. Numerical parity: Cached step logits match full-context forward pass logits within 1e-4.
3. Fast autoregressive generation execution without errors.
"""

import torch
import unittest
from soft_ltopology_lm.model import SoftLTopologicalLM


class TestKVCacheAutoregression(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embed_dim = 64
        self.num_layers = 2
        self.num_heads = 2
        self.hidden_dim = 128
        self.max_seq_len = 64
        self.model = SoftLTopologicalLM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            router_type="lukasiewicz",
            bounded_values=False,
        )
        self.model.eval()

    def test_cache_prefill_and_step(self):
        # Prefill with 4 tokens
        prompt = torch.tensor([[10, 25, 30, 45]])
        logits_prefill, _, metrics_prefill = self.model(prompt, use_cache=True)
        past_kv = metrics_prefill["past_key_values"]
        
        self.assertIsNotNone(past_kv)
        self.assertEqual(len(past_kv), self.num_layers)
        # Check shape of cached key: [B, H, S, D_h]
        self.assertEqual(past_kv[0][0].shape, (1, self.num_heads, 4, self.embed_dim // self.num_heads))
        
        # Single step with new token
        new_token = torch.tensor([[55]])
        logits_step, _, metrics_step = self.model(new_token, past_key_values=past_kv, use_cache=True)
        updated_kv = metrics_step["past_key_values"]
        
        self.assertEqual(updated_kv[0][0].shape, (1, self.num_heads, 5, self.embed_dim // self.num_heads))
        self.assertEqual(logits_step.shape, (1, 1, self.vocab_size))

    def test_numerical_parity_with_full_context(self):
        # Sequence of 5 tokens
        full_seq = torch.tensor([[12, 34, 56, 78, 90]])
        full_logits, _, _ = self.model(full_seq)
        expected_last_logits = full_logits[:, -1, :]
        
        # Now do prefill on first 4 tokens, then 1 step with 5th token
        prompt = full_seq[:, :4]
        _, _, prefill_metrics = self.model(prompt, use_cache=True)
        past_kv = prefill_metrics["past_key_values"]
        
        last_tok = full_seq[:, 4:5]
        step_logits, _, _ = self.model(last_tok, past_key_values=past_kv, use_cache=True)
        actual_last_logits = step_logits[:, 0, :]
        
        # Check logits match closely
        diff = (expected_last_logits - actual_last_logits).abs().max().item()
        self.assertLess(diff, 1e-4, f"Cached step logits deviate from full-context logits by {diff:.6f}")

    def test_fast_generation(self):
        prompt = torch.tensor([[5, 15, 25]])
        out_cached = self.model.generate(prompt, max_new_tokens=10, use_cache=True)
        out_uncached = self.model.generate(prompt, max_new_tokens=10, use_cache=False)
        self.assertEqual(out_cached.shape, (1, 13))
        self.assertEqual(out_uncached.shape, (1, 13))


if __name__ == "__main__":
    unittest.main()
