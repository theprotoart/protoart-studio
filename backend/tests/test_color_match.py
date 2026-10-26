import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import color_match


class ColorMatchTests(unittest.TestCase):
    def test_a_lighter_redraw_is_brought_back(self):
        rng = np.random.default_rng(0)
        base = (rng.random((256, 256, 3)) * 120 + 40).astype(np.float32)
        original = Image.fromarray(base.astype(np.uint8))
        redraw = np.clip(base * 1.12 + 18, 0, 255)  # the whole redraw came out lighter
        mask = Image.new("L", (256, 256), 0)
        ImageDraw.Draw(mask).ellipse([80, 80, 176, 176], fill=255)
        redraw[100:156, 100:156] = 200  # the new thing in the painted area
        fixed = np.asarray(color_match.match_to_original(Image.fromarray(redraw.astype(np.uint8)), original, mask), np.float32)
        ring = np.zeros((256, 256), bool)
        ring[70:186, 70:186] = True
        ring[90:166, 90:166] = False  # just outside / along the painted edge
        before = abs(redraw[ring] - base[ring]).mean()
        after = abs(fixed[ring] - base[ring]).mean()
        self.assertLess(after, before * 0.35)
        # the new thing is still clearly there (only the broad light changed)
        self.assertGreater(fixed[110:146, 110:146].mean(), base[110:146, 110:146].mean() + 25)

    def test_nothing_to_match(self):
        im = Image.new("RGB", (64, 64), (100, 100, 100))
        full = Image.new("L", (64, 64), 255)
        self.assertIs(color_match.match_to_original(im, im, full), im)

    def test_keep_tones_keeps_detail_and_restores_colour(self):
        rng = np.random.default_rng(1)
        orig = np.zeros((320, 320, 3), np.float32)
        orig[..., 0], orig[..., 1], orig[..., 2] = 180, 120, 60  # a warm painting
        detail = rng.normal(0, 12, (320, 320, 1)).astype(np.float32)
        enhanced = np.clip(np.full((320, 320, 3), 150, np.float32) + detail, 0, 255)  # greyer, with new texture
        out = np.asarray(color_match.keep_tones(Image.fromarray(enhanced.astype(np.uint8)),
                                                Image.fromarray(orig.astype(np.uint8))), np.float32)
        self.assertLess(abs(out[..., 0].mean() - 180), 4)
        self.assertLess(abs(out[..., 2].mean() - 60), 4)
        self.assertGreater(out[..., 1].std(), 6)  # the fine texture is still there


if __name__ == "__main__":
    unittest.main()
