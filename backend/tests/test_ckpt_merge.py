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


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
class MergeTests(unittest.TestCase):
    def setUp(self):
        import paths
        self.d = Path(tempfile.mkdtemp())
        self._m = paths.MODELS_DIR
        paths.MODELS_DIR = self.d / "models"

    def tearDown(self):
        import paths
        paths.MODELS_DIR = self._m

    def _save(self, name, v):
        import torch
        from safetensors.torch import save_file
        sd = {"model.diffusion_model.w": torch.full((3, 4), float(v)), "first_stage_model.vae": torch.full((2,), float(v)),
              "cond_stage_model.ids": torch.arange(5)}
        p = self.d / name
        save_file(sd, str(p))
        return str(p)

    def test_weighted_sum_and_add_difference(self):
        import ckpt_merge
        from safetensors.torch import load_file
        a, b, c = self._save("a.safetensors", 1), self._save("b.safetensors", 3), self._save("c.safetensors", 2)
        r = ckpt_merge.merge(a, b, 0.25, name="sum")
        out = load_file(r["path"])
        self.assertAlmostEqual(float(out["model.diffusion_model.w"][0, 0]), 1.5, places=3)
        self.assertAlmostEqual(float(out["first_stage_model.vae"][0]), 1.0, places=3)  # the VAE stays model A's
        self.assertEqual(out["cond_stage_model.ids"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(str(out["model.diffusion_model.w"].dtype), "torch.float16")
        r = ckpt_merge.merge(a, b, 1.0, mode="add", c=c, name="add")
        self.assertAlmostEqual(float(load_file(r["path"])["model.diffusion_model.w"][1, 1]), 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
