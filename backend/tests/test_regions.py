import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regions

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


class RegionLayoutTests(unittest.TestCase):
    def test_columns_and_rows(self):
        self.assertEqual(regions.parse_layout("cols", "1,1", 2), [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)])
        rows = regions.parse_layout("rows", "1,3", 2)
        self.assertAlmostEqual(rows[0][3], 0.25)
        self.assertEqual(rows[1][1], 0.25)

    def test_default_ratios_and_grid(self):
        self.assertEqual(len(regions.parse_layout("cols", "", 3)), 3)
        g = regions.parse_layout("grid", "1,1;1", 3)
        self.assertEqual(g, [(0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 1.0)])

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
    def test_masks_sum_to_one(self):
        import torch
        m = regions.masks(regions.parse_layout("cols", "1,2", 2), 16, 24, "cpu", torch.float32)
        self.assertTrue(torch.allclose(m.sum(0), torch.ones(16, 24), atol=1e-5))
        self.assertGreater(float(m[0, 8, 2]), 0.99)
        self.assertGreater(float(m[1, 8, 20]), 0.99)


if __name__ == "__main__":
    unittest.main()
