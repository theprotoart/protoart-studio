"""Object selection (GrabCut): finds a plain object inside the box."""
import base64
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SelectObjectTests(unittest.TestCase):
    def test_finds_disc(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("no OpenCV")
        import select_object
        img = Image.new("RGB", (200, 160), (40, 120, 200))
        rng = np.random.default_rng(0)
        noise = (rng.random((160, 200, 3)) * 30).astype(np.uint8)
        img = Image.fromarray(np.clip(np.asarray(img) + noise, 0, 255).astype(np.uint8))
        ImageDraw.Draw(img).ellipse((70, 50, 130, 110), fill=(230, 60, 40))
        url, found = select_object.select(img, (0.25, 0.2, 0.5, 0.6), 200, 160)
        self.assertTrue(found)
        m = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))[..., 3] > 0
        self.assertTrue(m[80, 100])          # centre of the disc picked
        self.assertFalse(m[10, 10])          # background outside the box not picked
        self.assertFalse(m[60, 55])          # background inside the box not picked
        area = m.sum()
        self.assertLess(abs(area - 3.1416 * 30 * 30) / (3.1416 * 30 * 30), 0.25)


if __name__ == "__main__":
    unittest.main()
