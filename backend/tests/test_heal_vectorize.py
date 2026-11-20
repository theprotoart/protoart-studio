"""🩹 Remove (spot healing) and ✒️ Vectorize: no model download needed for these tests."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import cv2  # noqa: F401
    import numpy as np
    from PIL import Image, ImageDraw
    HAVE_CV = True
except Exception:  # pragma: no cover
    HAVE_CV = False


@unittest.skipUnless(HAVE_CV, "needs OpenCV")
class HealTests(unittest.TestCase):
    def picture(self):
        img = Image.new("RGB", (300, 200), (40, 120, 200))
        ImageDraw.Draw(img).rectangle((120, 70, 170, 120), fill=(250, 20, 20))  # a red thing to remove
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rectangle((115, 65, 175, 125), fill=255)
        return img, mask

    def test_quick_fill_removes_the_object_and_keeps_the_rest(self):
        from postprocess import heal
        img, mask = self.picture()
        out = np.asarray(heal.heal(img, mask, method="quick")).astype(int)
        self.assertLess(out[95, 145][0], 120)                  # no red left in the middle
        self.assertEqual(list(out[10, 10]), [40, 120, 200])     # outside: untouched

    def test_lama_path_with_a_stand_in_model(self):
        from postprocess import heal

        class I:
            def __init__(self, n):
                self.name = n

        class FakeLama:  # same inputs and outputs as the real export (image 0..1, mask 0/1 -> 0..255)
            def get_inputs(self):
                return [I("image"), I("mask")]

            def run(self, _o, feed):
                im, m = feed["image"], feed["mask"]
                assert im.shape == (1, 3, 512, 512) and m.shape == (1, 1, 512, 512)
                return [(im * (1 - m) + np.array([40, 120, 200], np.float32).reshape(1, 3, 1, 1) / 255 * m) * 255]
        img, mask = self.picture()
        out = np.asarray(heal.heal(img, mask, session=FakeLama())).astype(int)
        self.assertTrue(all(abs(a - b) <= 6 for a, b in zip(out[95, 145], [40, 120, 200])))
        self.assertEqual(list(out[10, 10]), [40, 120, 200])

    def test_nothing_painted(self):
        from postprocess import heal
        img, _ = self.picture()
        with self.assertRaises(ValueError):
            heal.heal(img, Image.new("L", img.size, 0), method="quick")


@unittest.skipUnless(HAVE_CV, "needs OpenCV")
class VectorizeTests(unittest.TestCase):
    def test_flat_shapes_become_paths(self):
        from postprocess import vectorize
        img = Image.new("RGB", (400, 300), (255, 210, 90))
        d = ImageDraw.Draw(img)
        d.ellipse((60, 60, 200, 200), fill=(30, 30, 30))
        d.rectangle((240, 80, 360, 240), fill=(40, 110, 220))
        svg, st = vectorize.vectorize(img, colors=3, detail="medium")
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertIn('viewBox="0 0 400 300"', svg)
        self.assertGreaterEqual(st["shapes"], 2)
        self.assertEqual(svg.count("<path"), 2)
        self.assertTrue(svg.index("<rect") < svg.index("<path"))


def test_heal_long_thin_mark_is_worked_along():
    import numpy as np
    from postprocess import heal
    m = np.zeros((1080, 1920), bool)
    m[900:960, 200:1700] = True  # a watermark line along the bottom
    m[100:150, 100:160] = True   # and a separate spot
    parts = heal._parts(m)
    assert len(parts) == 2 and heal._thickness(parts[0]) == 60
    wins = heal._windows(parts[0])
    assert len(wins) >= 4 and all(x1 - x0 == 512 and y1 - y0 == 512 for x0, y0, x1, y1 in wins)
    covered = np.zeros_like(m)
    for x0, y0, x1, y1 in wins:
        covered[y0:y1, x0:x1] = True
    assert covered[parts[0]].all()
