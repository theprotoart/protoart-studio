"""Batch edit & export and Press (smallest file that looks the same): sizes, formats, names, privacy."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch_export  # noqa: E402
import press  # noqa: E402


def photo(w=320, h=240):
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([128 + 90 * np.sin(x / 23), 120 + 70 * np.cos(y / 17), 140 + 60 * np.sin((x + y) / 31)], -1)
    return Image.fromarray(np.clip(base + rng.normal(0, 9, base.shape), 0, 255).astype("uint8"))


class SizeAndNameTests(unittest.TestCase):
    def test_target_size(self):
        self.assertEqual(batch_export.target_size(4000, 3000, {"mode": "long", "value": 2000}), (2000, 1500))
        self.assertEqual(batch_export.target_size(4000, 3000, {"mode": "short", "value": 1500}), (2000, 1500))
        self.assertEqual(batch_export.target_size(400, 300, {"mode": "long", "value": 2000}), (400, 300))  # no enlarging
        self.assertEqual(batch_export.target_size(400, 300, {"mode": "long", "value": 800, "no_upscale": False}), (800, 600))
        self.assertEqual(batch_export.target_size(1000, 500, {"mode": "fit", "w": 500, "h": 500}), (500, 250))
        self.assertEqual(batch_export.target_size(1000, 500, {"mode": "percent", "value": 50}), (500, 250))

    def test_file_names_are_safe(self):
        n = batch_export.file_name("{name}_{n}/../{model}", {"name": "a:b", "n": 7, "model": "dream?shaper"})
        self.assertEqual(n, "a_b_007_.._dream_shaper")
        self.assertNotIn("/", n)
        self.assertEqual(batch_export.file_name("{nope}", {}), "{nope}")


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "pic.png")
        photo().save(self.src)
        self.out = os.path.join(self.tmp.name, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_formats_dpi_and_no_overwrite(self):
        settings = {"prompt": "a secret prompt", "model": "local:/Users/me/x.safetensors", "seed": 5}
        for fmt in ("png", "jpeg", "webp", "tiff"):
            r = batch_export.export_one(self.src, self.out, {"format": fmt, "dpi": 300}, {"name": "pic"}, settings)
            with Image.open(r["path"]) as im:
                self.assertEqual(im.size, (320, 240))
                blob = open(r["path"], "rb").read()
                self.assertNotIn(b"secret prompt", blob)  # details are left out unless you keep them
        again = batch_export.export_one(self.src, self.out, {"format": "png"}, {"name": "pic"})
        self.assertTrue(again["name"].startswith("pic-"))

    def test_keep_details_without_paths(self):
        settings = {"prompt": "a cabin", "model": "local:/Users/me/x.safetensors", "seed": 5}
        r = batch_export.export_one(self.src, self.out, {"format": "png", "keep_meta": True}, {"name": "k"}, settings)
        with Image.open(r["path"]) as im:
            self.assertIn("a cabin", im.info.get("parameters", ""))
            self.assertNotIn("/Users/me", json.dumps({k: v for k, v in im.info.items() if isinstance(v, str)}))

    def test_press_jpeg_and_png(self):
        r = batch_export.export_one(self.src, self.out, {"format": "jpeg", "compress": "balanced"}, {"name": "p"})
        self.assertIn("pressed", r["detail"])
        plain = batch_export.export_one(self.src, self.out, {"format": "jpeg", "quality": 96}, {"name": "q"})
        self.assertLess(r["bytes"], plain["bytes"])
        same = batch_export.export_one(self.src, self.out, {"format": "same", "compress": "gentle"}, {"name": "s"})
        self.assertEqual(same["format"], "png")


class PressTests(unittest.TestCase):
    def test_ssim_identical_and_different(self):
        a = press.luma(photo())
        self.assertAlmostEqual(press.compare(a, a)["mean"], 1.0, places=6)
        b = a.copy()
        b[:64, :64] = 0
        self.assertLess(press.compare(a, b)["worst_region"], 0.5)

    def test_png_palette_is_exact(self):
        arr = np.zeros((40, 40, 4), np.uint8)
        arr[..., 3] = 255
        arr[:20, :, 0] = 200
        arr[:5, :5, 3] = 0
        data, how = press.smallest_png(Image.fromarray(arr, "RGBA"), "gentle")
        back = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        self.assertTrue(np.array_equal(back[5:], arr[5:]))
        self.assertEqual(how, "lossless")


if __name__ == "__main__":
    unittest.main()
