import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


class Evil:
    """Opening a file with this inside the ordinary way would create a marker file."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


class FakeCallback:  # like pytorch_lightning's ModelCheckpoint: a class this app doesn't have
    def __init__(self):
        self.best = 1.0


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
class CkptConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        import paths
        self._models = paths.MODELS_DIR
        paths.MODELS_DIR = self.tmp / "models"

    def tearDown(self):
        import paths
        paths.MODELS_DIR = self._models

    def _write(self, name, extra):
        import torch
        sd = {"model.diffusion_model.a.weight": torch.randn(4, 4), "first_stage_model.b": torch.ones(3)}
        p = self.tmp / name
        torch.save({"state_dict": sd, **extra}, p)
        return p, sd

    def test_lightning_style_file_converts(self):
        import ckpt_convert
        from safetensors.torch import load_file
        p, sd = self._write("old.ckpt", {"callbacks": FakeCallback(), "epoch": 3})
        r = ckpt_convert.convert(p, half=True)
        out = load_file(r["path"])
        self.assertEqual(set(out), set(sd))
        self.assertEqual(out["model.diffusion_model.a.weight"].dtype.__str__(), "torch.float16")
        self.assertTrue(p.exists())  # the original is left as it was

    def test_code_inside_is_never_run(self):
        import ckpt_convert
        marker = self.tmp / "ran"
        p, sd = self._write("evil.ckpt", {"payload": Evil(marker)})
        ckpt_convert.convert(p, half=False)
        self.assertFalse(marker.exists())

    def test_find(self):
        import ckpt_convert
        self.assertEqual(ckpt_convert.find_ckpts([str(self.tmp)]), [])  # tiny test files are under 1 GB


if __name__ == "__main__":
    unittest.main()
