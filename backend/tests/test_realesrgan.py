"""⏩ Instant upscalers: the Real-ESRGAN networks have the layer names and counts of Real-ESRGAN's checkpoints
(checked against the released files: 702 / 192 / 53 / 101 tensors; x2plus takes 12 channels) and make pictures 4× (or
2×) larger; the community ESRGAN models (AnimeSharp, Nomos8kSC) use the old "model.N" layout (702 tensors)."""
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
        g = r.SRVGGNetCompact(32).state_dict()
        self.assertEqual(len(g), 101)
        self.assertEqual(tuple(g["body.66.weight"].shape), (48, 64, 3, 3))
        x2 = r.RRDBNet(23, scale=2).state_dict()
        self.assertEqual(len(x2), 702)
        self.assertEqual(tuple(x2["conv_first.weight"].shape), (64, 12, 3, 3))
        from postprocess.esrgan_model import ESRGAN4x
        self.assertEqual(len(ESRGAN4x().state_dict()), 702)

    def test_four_times_larger(self):
        from postprocess import realesrgan as r
        x = torch.rand(1, 3, 8, 12)
        with torch.no_grad():
            self.assertEqual(tuple(r.RRDBNet(1)(x).shape), (1, 3, 32, 48))
            self.assertEqual(tuple(r.SRVGGNetCompact(2)(x).shape), (1, 3, 32, 48))
            self.assertEqual(tuple(r.RRDBNet(1, scale=2)(x).shape), (1, 3, 16, 24))
            odd = torch.rand(1, 3, 7, 9)  # odd sizes are padded to fold, then cropped back
            self.assertEqual(tuple(r.RRDBNet(1, scale=2)(odd).shape), (1, 3, 14, 18))

    def test_downloads_are_pinned(self):
        from postprocess import realesrgan as r
        for m in r.MODELS.values():
            self.assertTrue(m["url"].startswith(("https://github.com/xinntao/Real-ESRGAN/releases/download/",
                                                 "https://github.com/Phhofm/models/releases/download/",
                                                 "https://huggingface.co/Kim2091/AnimeSharp/resolve/",
                                                 "https://huggingface.co/Kim2091/UltraSharpV2/resolve/")))
            self.assertEqual(len(m["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
