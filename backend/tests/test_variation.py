import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
class VariationNoiseTests(unittest.TestCase):
    def test_strength_zero_is_the_normal_noise(self):
        from engine import variation_noise
        a = variation_noise(42, 7, 0.0, (1, 4, 8, 8), torch.float32)
        b = torch.randn((1, 4, 8, 8), generator=torch.Generator("cpu").manual_seed(42))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_small_strength_stays_close_and_keeps_scale(self):
        from engine import variation_noise
        base = variation_noise(42, 7, 0.0, (1, 4, 32, 32), torch.float32)
        v = variation_noise(42, 7, 0.1, (1, 4, 32, 32), torch.float32)
        self.assertGreater(torch.nn.functional.cosine_similarity(base.flatten(), v.flatten(), dim=0).item(), 0.95)
        self.assertAlmostEqual(v.std().item(), base.std().item(), delta=0.05)

    def test_full_strength_is_the_other_seed(self):
        from engine import variation_noise
        a = variation_noise(42, 7, 1.0, (1, 4, 8, 8), torch.float32)
        b = torch.randn((1, 4, 8, 8), generator=torch.Generator("cpu").manual_seed(7))
        self.assertTrue(torch.allclose(a, b, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
