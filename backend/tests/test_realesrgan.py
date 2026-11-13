"""⏩ Instant upscalers: the Real-ESRGAN networks have the layer names and counts of Real-ESRGAN's checkpoints
(checked against the released files: 702 / 192 / 53 tensors) and make pictures 4× larger."""
import unittest

try:
    import torch
except ImportError:  # the app's own Python has torch; a bare test machine may not
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class RealESRGANNets(unittest.TestCase):
    def test_layer_names_match_the_checkpoints(self):
        from postprocess import realesrgan as r
        x4 = r.RRDBNet(23).state_dict()
        self.assertEqual(len(x4), 702)
        self.assertIn("body.22.rdb3.conv5.weight", x4)
        self.assertEqual(len(r.RRDBNet(6).state_dict()), 192)
        v3 = r.SRVGGNetCompact(16).state_dict()
        self.assertEqual(len(v3), 53)
        self.assertEqual(tuple(v3["body.34.weight"].shape), (48, 64, 3, 3))

    def test_four_times_larger(self):
        from postprocess import realesrgan as r
        x = torch.rand(1, 3, 8, 12)
        with torch.no_grad():
            self.assertEqual(tuple(r.RRDBNet(1)(x).shape), (1, 3, 32, 48))
            self.assertEqual(tuple(r.SRVGGNetCompact(2)(x).shape), (1, 3, 32, 48))

    def test_downloads_are_pinned(self):
        from postprocess import realesrgan as r
        for m in r.MODELS.values():
            self.assertTrue(m["url"].startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/"))
            self.assertEqual(len(m["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
