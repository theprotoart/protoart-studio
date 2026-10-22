import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anima


class AnimaRatingTests(unittest.TestCase):
    def test_safe_added_when_no_rating(self):
        self.assertEqual(anima.with_rating("1girl, red hair"), "safe, 1girl, red hair")
        self.assertEqual(anima.with_rating(""), "safe, ")

    def test_own_rating_kept(self):
        for r in ("safe", "Sensitive", "nsfw", "explicit"):
            p = f"scenery, {r}, lake"
            self.assertEqual(anima.with_rating(p), p)

    def test_word_inside_a_tag_is_not_a_rating(self):
        self.assertTrue(anima.with_rating("safe house, night").startswith("safe, "))

    def test_negative_gets_other_ratings(self):
        neg = anima.safe_negative(anima.with_rating("cat"), "blurry")
        self.assertEqual(neg, "blurry, explicit, nsfw, sensitive")
        self.assertEqual(anima.safe_negative("safe, cat", ""), "explicit, nsfw, sensitive")
        self.assertEqual(anima.safe_negative("explicit, cat", "blurry"), "blurry")


class AnimaVaeFixTests(unittest.TestCase):
    def test_conv_as_2d_matches_conv3d(self):
        try:
            import torch
            from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
        except Exception:
            self.skipTest("torch / diffusers not installed")
        import types
        torch.manual_seed(0)
        for k, stride in ((3, 1), (1, 1), (3, 2)):
            conv = QwenImageCausalConv3d(8, 6, k, stride=stride, padding=k // 2)
            x = torch.randn(1, 8, 1 if stride == 1 else 4, 12, 12)
            ref = conv(x)
            conv.forward = types.MethodType(anima._conv3d_as_2d, conv)
            self.assertTrue(torch.allclose(conv(x), ref, atol=1e-5), (k, stride))
            cache = torch.randn(1, 8, 2, 12, 12)
            if k == 3:
                c2 = QwenImageCausalConv3d(8, 6, 3, padding=1)
                ref2 = c2(x[:, :, :1], cache)
                c2.forward = types.MethodType(anima._conv3d_as_2d, c2)
                self.assertTrue(torch.allclose(c2(x[:, :, :1], cache), ref2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
