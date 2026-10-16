"""Opening any picture: PNG/JPEG direct, HEIC/TIFF/RAW through macOS's own reader (sips), cached."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import pictures  # noqa: E402


class PictureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (paths.HOME, pictures.SIPS)
        paths.HOME = Path(self.tmp.name) / "home"
        self.png = os.path.join(self.tmp.name, "a.png")
        Image.new("RGB", (20, 10), "red").save(self.png)
        # a stand-in for /usr/bin/sips: "converts" by copying a PNG (records each call)
        self.calls = os.path.join(self.tmp.name, "calls")
        fake = os.path.join(self.tmp.name, "sips")
        with open(fake, "w") as f:
            f.write(f"#!/bin/sh\necho x >> {self.calls}\nfor last; do :; done\ncp {self.png} \"$last\"\n")
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
        pictures.SIPS = fake

    def tearDown(self):
        paths.HOME, pictures.SIPS = self.old
        self.tmp.cleanup()

    def test_direct_formats_untouched(self):
        self.assertEqual(pictures.prepare(self.png), self.png)

    def test_heic_converted_once(self):
        heic = os.path.join(self.tmp.name, "IMG_0001.HEIC")
        Path(heic).write_bytes(b"not really heic")
        out = pictures.prepare(heic)
        self.assertTrue(out.endswith(".png") and os.path.isfile(out))
        self.assertEqual(Image.open(out).size, (20, 10))
        self.assertEqual(pictures.prepare(heic), out)  # cached
        self.assertEqual(len(Path(self.calls).read_text().split()), 1)

    def test_refuses_other_files(self):
        txt = os.path.join(self.tmp.name, "notes.txt")
        Path(txt).write_text("hi")
        with self.assertRaises(pictures.PictureError):
            pictures.prepare(txt)
        with self.assertRaises(pictures.PictureError):
            pictures.prepare("relative.png")
        pictures.SIPS = "/nonexistent/sips"
        raw = os.path.join(self.tmp.name, "shot.dng")
        Path(raw).write_bytes(b"x")
        with self.assertRaises(pictures.PictureError):
            pictures.prepare(raw)


if __name__ == "__main__":
    unittest.main()
