"""VAEs and ControlNets from Civitai, your own ControlNets, and safe-for-work mode."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import civitai  # noqa: E402


def _ver(base, name="x.safetensors"):
    return {"id": 9, "baseModel": base, "files": [{"name": name, "sizeKB": 1000, "type": "Model", "primary": True}]}


class CivitaiKindsTests(unittest.TestCase):
    def test_vae(self):
        d = civitai.describe({"id": 1, "name": "vae-ft-mse", "type": "VAE"}, _ver("SD 1.5"))
        self.assertEqual(d["kind"], "vae")
        self.assertTrue(d["ok"], d["notes"])

    def test_controlnet(self):
        ok = civitai.describe({"id": 2, "name": "QR", "type": "Controlnet"}, _ver("SD 1.5"))
        self.assertEqual(ok["kind"], "controlnet")
        self.assertTrue(ok["ok"], ok["notes"])
        no = civitai.describe({"id": 3, "name": "QR XL", "type": "Controlnet"}, _ver("SDXL 1.0"))
        self.assertFalse(no["ok"])


class UserControlnetTests(unittest.TestCase):
    def test_user_types_and_map(self):
        import paths
        import controlnet
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            old = paths.CONTROLNET_DIR
            paths.CONTROLNET_DIR = Path(d)
            try:
                Path(d, "qrMonster.safetensors").write_bytes(b"0" * 10)
                ids = [t["id"] for t in controlnet.list_types()]
                self.assertIn("user:qrMonster", ids)
                self.assertTrue(controlnet.is_known("user:qrMonster"))
                self.assertFalse(controlnet.is_known("user:nope"))
                img = Image.new("RGB", (40, 30), (200, 10, 10))
                out = controlnet.make_map("user:qrMonster", img, 64, 48)
                self.assertEqual(out.size, (64, 48))
                self.assertEqual(out.getpixel((5, 5)), (200, 10, 10))  # the picture as it is
            finally:
                paths.CONTROLNET_DIR = old


class VaeTests(unittest.TestCase):
    def test_family_from_name(self):
        import paths
        import vaes
        with tempfile.TemporaryDirectory() as d:
            old = paths.VAE_DIR
            paths.VAE_DIR = Path(d)
            try:
                for n in ("vae-ft-mse-840000.safetensors", "sdxl_vae.safetensors"):
                    with open(os.path.join(d, n), "wb") as f:
                        f.truncate(90_000_000)
                fams = {v["name"]: v["family"] for v in vaes.list_vaes()}
                self.assertEqual(fams, {"vae-ft-mse-840000": "sd15", "sdxl_vae": "sdxl"})
            finally:
                paths.VAE_DIR = old



class SfwTests(unittest.TestCase):
    def test_blocks(self):
        try:
            import bee_backend as bb
            from engine import UserError
        except Exception as e:
            self.skipTest(str(e))
        bb.settings["sfw_mode"] = True
        try:
            with self.assertRaises(UserError):
                bb.sfw_check("generate", {"prompt": "a nude woman on a beach", "model": "sd15"})
            with self.assertRaises(UserError):
                bb.sfw_check("generate", {"prompt": "a cat", "model": "sd15", "loras": [{"name": "hentai_style_v2"}]})
            bb.sfw_check("generate", {"prompt": "a cat on a sofa", "model": "sd15"})  # fine
            bb.settings["sfw_mode"] = False
            bb.sfw_check("generate", {"prompt": "a nude woman on a beach", "model": "sd15"})  # off: not checked
        finally:
            bb.settings["sfw_mode"] = False


if __name__ == "__main__":
    unittest.main()
