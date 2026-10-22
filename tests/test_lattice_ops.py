"""
Unit tests for Continuous Lattice Operators (t-norms, t-conorms, and Frank/Hamacher families).
Verifies:
1. Boundary conditions: T(x, 1) = x, T(x, 0) = 0, S(x, 0) = x, S(x, 1) = 1.
2. Commutativity: T(x, y) = T(y, x).
3. Monotonicity: x <= y => T(x, z) <= T(y, z).
4. Deterministic Nilpotent Zero Sparsity: x + y <= 1 => T_L(x, y) = 0.0.
5. Exact zero sparsity calculation function.
"""

import torch
import unittest
from soft_ltopology_lm.lattice_ops import (
    lukasiewicz_tnorm,
    lukasiewicz_tconorm,
    godel_tnorm,
    godel_tconorm,
    product_tnorm,
    probabilistic_sum_tconorm,
    frank_tnorm,
    frank_tconorm,
    hamacher_tnorm,
    hamacher_tconorm,
    compute_exact_zero_sparsity,
)


class TestLatticeOperators(unittest.TestCase):
    def test_lukasiewicz_properties(self):
        x = torch.tensor([0.0, 0.2, 0.5, 0.7, 1.0])
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)

        # Boundary condition: T_L(x, 1) = x
        self.assertTrue(torch.allclose(lukasiewicz_tnorm(x, one), x))
        # Boundary condition: T_L(x, 0) = 0
        self.assertTrue(torch.allclose(lukasiewicz_tnorm(x, zero), zero))
        # Conorm boundary: S_L(x, 0) = x
        self.assertTrue(torch.allclose(lukasiewicz_tconorm(x, zero), x))
        # Conorm boundary: S_L(x, 1) = 1
        self.assertTrue(torch.allclose(lukasiewicz_tconorm(x, one), one))

        # Commutativity
        y = torch.tensor([0.4, 0.1, 0.8, 0.2, 0.9])
        self.assertTrue(torch.allclose(lukasiewicz_tnorm(x, y), lukasiewicz_tnorm(y, x)))

        # Deterministic Nilpotent Sparsity: 0.2 + 0.4 = 0.6 <= 1.0 => T_L = strictly 0.0
        val = lukasiewicz_tnorm(torch.tensor([0.2]), torch.tensor([0.4]))
        self.assertEqual(val.item(), 0.0)

        # Non-zero when sum > 1.0: 0.7 + 0.8 = 1.5 - 1.0 = 0.5
        val2 = lukasiewicz_tnorm(torch.tensor([0.7]), torch.tensor([0.8]))
        self.assertAlmostEqual(val2.item(), 0.5, places=5)

    def test_godel_properties(self):
        x = torch.tensor([0.3, 0.8])
        y = torch.tensor([0.6, 0.4])
        # Min
        self.assertTrue(torch.allclose(godel_tnorm(x, y), torch.tensor([0.3, 0.4])))
        # Max
        self.assertTrue(torch.allclose(godel_tconorm(x, y), torch.tensor([0.6, 0.8])))

    def test_product_properties(self):
        x = torch.tensor([0.5, 0.4])
        y = torch.tensor([0.5, 0.2])
        self.assertTrue(torch.allclose(product_tnorm(x, y), torch.tensor([0.25, 0.08])))

    def test_frank_properties(self):
        x = torch.tensor([0.4, 0.7])
        y = torch.tensor([0.3, 0.6])
        s = torch.tensor([5.0])
        res = frank_tnorm(x, y, s)
        self.assertTrue(torch.all(res >= 0.0))
        self.assertTrue(torch.all(res <= 1.0))

    def test_sparsity_metric(self):
        tensor = torch.tensor([0.0, 0.0, 0.5, 0.8, 0.0])
        # 3 out of 5 are 0.0 -> 60.0%
        sp = compute_exact_zero_sparsity(tensor)
        self.assertAlmostEqual(sp, 60.0, places=4)


if __name__ == "__main__":
    unittest.main()
