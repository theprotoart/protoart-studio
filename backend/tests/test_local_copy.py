import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_copy
import paths


class KeepCopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.drive, self.models = root / "drive", root / "models"
        self.drive.mkdir()
        self.src = self.drive / "photon_v1.safetensors"
        self.src.write_bytes(os.urandom(3 * local_copy.CHUNK + 123))
        (self.drive / "photon_v1.civitai.info").write_text('{"vae": "x"}')
        (self.drive / "photon_v1.preview.png").write_bytes(b"png")
        (self.drive / "photon_v1_other.safetensors").write_bytes(b"not a sidecar")
        self.patches = [mock.patch.object(paths, "MODELS_DIR", self.models),
                        mock.patch.object(local_copy, "on_external_drive", lambda p: True),
                        mock.patch.object(local_copy, "MARGIN", 0),
                        mock.patch.dict(local_copy.KEPT, clear=True)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_copies_without_touching_the_original(self):
        before = (hashlib.md5(self.src.read_bytes()).hexdigest(), self.src.stat().st_mtime)
        seen = []
        r = local_copy.copy(self.src, lambda d, t: seen.append(d))
        self.assertEqual(r["id"], "file:photon_v1.safetensors")
        self.assertEqual(hashlib.md5((self.models / self.src.name).read_bytes()).hexdigest(), before[0])
        self.assertEqual((hashlib.md5(self.src.read_bytes()).hexdigest(), self.src.stat().st_mtime), before)
        self.assertTrue((self.models / "photon_v1.civitai.info").exists())
        self.assertTrue((self.models / "photon_v1.preview.png").exists())
        self.assertFalse((self.models / "photon_v1_other.safetensors").exists())
        self.assertFalse(list(self.models.glob("*.part")))
        self.assertEqual(seen[-1], self.src.stat().st_size)
        self.assertEqual(local_copy.copy_id(self.src), r["id"])
        (self.models / self.src.name).unlink()  # the copy moved to the Trash: the drive's file is used again
        self.assertIsNone(local_copy.copy_id(self.src))

    def test_not_enough_space(self):
        with mock.patch.object(local_copy.shutil, "disk_usage", lambda p: mock.Mock(free=10)):
            c = local_copy.check(self.src)
            self.assertFalse(c["ok"])
            self.assertIn("Not enough free space", c["why"])
            with self.assertRaises(ValueError):
                local_copy.copy(self.src)
        self.assertFalse(self.models.joinpath(self.src.name).exists())

    def test_never_overwrites_a_different_file(self):
        self.models.mkdir()
        (self.models / self.src.name).write_bytes(b"someone else's model")
        self.assertFalse(local_copy.check(self.src)["ok"])
        with self.assertRaises(ValueError):
            local_copy.copy(self.src)
        self.assertEqual((self.models / self.src.name).read_bytes(), b"someone else's model")

    def test_cancel_leaves_nothing(self):
        with self.assertRaises(InterruptedError):
            local_copy.copy(self.src, cancelled=lambda: True)
        self.assertFalse(list(self.models.glob("*")))
        self.assertTrue(self.src.exists())

    def test_only_drive_safetensors(self):
        ck = self.drive / "old.ckpt"
        ck.write_bytes(b"x")
        self.assertFalse(local_copy.check(ck)["ok"])
        with mock.patch.object(local_copy, "on_external_drive", lambda p: False):
            self.assertIn("already on this Mac", local_copy.check(self.src)["why"])


if __name__ == "__main__":
    unittest.main()
