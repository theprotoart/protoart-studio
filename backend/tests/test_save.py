"""Saving images as PNG/JPEG with settings embedded, and reading them back as external sources."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

import paths  # noqa: E402


@unittest.skipIf(Image is None, "Pillow not installed")
class SaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        paths.IMAGES_DIR = Path(self.tmp.name)
        from engine import Engine
        self.eng = Engine(library=None, emit=lambda m: None)
        self.settings = {"prompt": "a red fox", "negative_prompt": "blurry", "steps": 8, "cfg": 1.0,
                         "sampler": "flow_euler", "seed": 42, "width": 64, "height": 64,
                         "model": "ernie_turbo_q6", "loras": [{"name": "fur", "weight": 0.7}]}
        self.img = Image.new("RGB", (64, 64), (200, 60, 30))

    def tearDown(self):
        self.tmp.cleanup()

    def test_png_keeps_settings(self):
        p = self.eng._save(self.img, self.settings)
        self.assertEqual(p.suffix, ".png")
        with Image.open(p) as im:
            self.assertEqual(json.loads(im.text["beestudio"])["seed"], 42)
            self.assertIn("<lora:fur:0.7>", im.text["parameters"])
        src = self.eng._external_source(p)
        self.assertEqual(src["prompt"], "a red fox")
        self.assertEqual(src["loras"], [{"name": "fur", "weight": 0.7}])

    def test_jpeg_option(self):
        self.eng.output_format = ("jpeg", 85)
        p = self.eng._save(self.img, self.settings, suffix="enhanced")
        self.assertEqual(p.suffix, ".jpg")
        self.assertTrue(p.name.endswith("_42_enhanced.jpg"))
        with Image.open(p) as im:
            self.assertEqual(im.format, "JPEG")
            comment = im.info.get("comment", b"").decode("utf-8")
        self.assertIn("a red fox", comment)
        self.assertIn('"seed": 42', comment)

    def test_thumbnails(self):
        import thumbs
        paths.HOME = Path(self.tmp.name)
        big = Image.new("RGB", (1024, 768), (10, 120, 200))
        src = Path(self.tmp.name) / "big.png"
        big.save(src)
        recs = thumbs.ensure([{"id": 7, "path": str(src)}, {"id": 8, "path": "/missing.png"}])
        self.assertTrue(recs[0]["thumb"].endswith("thumbs/7.jpg"))
        self.assertIsNone(recs[1]["thumb"])
        with Image.open(recs[0]["thumb"]) as im:
            self.assertEqual(max(im.size), thumbs.SIZE)
        thumbs.remove([7])
        self.assertFalse(Path(recs[0]["thumb"]).exists())

    def test_external_source_without_metadata(self):
        p = Path(self.tmp.name) / "photo.jpg"
        self.img.save(p)
        src = self.eng._external_source(p)
        self.assertIsNone(src["id"])
        self.assertEqual(src["prompt"], "")


if __name__ == "__main__":
    unittest.main()
