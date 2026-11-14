"""A picture dropped from a temporary place (a screenshot's floating thumbnail, a browser, Photos) is copied into the
app's imports folder, so it can't vanish half way through (it did: "Couldn't open that file" in Clear it)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import paths
import pictures


class TempPictures(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(dir=str(Path.home()))  # not under /tmp, which counts as temporary
        self.addCleanup(self.d.cleanup)
        self.home = Path(self.d.name) / "home"
        p = mock.patch.object(paths, "HOME", self.home)
        p.start(); self.addCleanup(p.stop)

    def temp_png(self):
        d = Path(self.d.name) / "var" / "folders" / "xx" / "T" / "TemporaryItems" / "NSIRD_screencaptureui_1"
        d.mkdir(parents=True)
        f = d / "Screenshot 2026-11-14.png"
        f.write_bytes(b"\x89PNG fake picture")
        return f

    def test_temporary_picture_is_kept(self):
        f = self.temp_png()
        out = pictures.prepare(str(f))
        self.assertTrue(out.startswith(str(self.home / "imports")))
        os.remove(f)  # macOS removes the original…
        self.assertEqual(Path(out).read_bytes(), b"\x89PNG fake picture")  # …the app's copy stays

    def test_same_picture_reuses_the_copy(self):
        f = self.temp_png()
        self.assertEqual(pictures.prepare(str(f)), pictures.prepare(str(f)))

    def test_normal_pictures_stay_where_they_are(self):
        f = Path(self.d.name) / "Pictures" / "comic.png"
        f.parent.mkdir()
        f.write_bytes(b"x")
        self.assertEqual(pictures.prepare(str(f)), str(f))

    def test_gone_file_says_what_happened(self):
        with self.assertRaises(pictures.PictureError) as cm:
            pictures.prepare(str(Path(self.d.name) / "image_0.png"))
        self.assertIn("image_0.png", str(cm.exception))
        self.assertIn("Add the picture again", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
