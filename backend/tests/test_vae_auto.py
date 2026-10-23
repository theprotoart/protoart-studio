import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import civitai
import vaes


def fake_safetensors(path, keys):
    header = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


ROWS = [{"name": "vae-ft-mse-840000-ema-pruned", "family": "sd15"}, {"name": "sdxl-vae-fp16-fix", "family": "sdxl"},
        {"name": "klF8Anime2", "family": "sd15"}]


class AutoVaeTests(unittest.TestCase):
    def test_sdxl_uses_fp16_fix(self):
        name, why = vaes.auto_for(None, "sdxl", ROWS)
        self.assertEqual(name, "sdxl-vae-fp16-fix")
        self.assertIsNone(vaes.auto_for(None, "sdxl", ROWS[:1])[0])  # not downloaded: the model's own

    def test_sd15_with_and_without_its_own_vae(self):
        with tempfile.TemporaryDirectory() as d:
            own, bare = Path(d) / "own.safetensors", Path(d) / "bare.safetensors"
            fake_safetensors(own, ["model.diffusion_model.x", "first_stage_model.decoder.conv_in.weight"])
            fake_safetensors(bare, ["model.diffusion_model.x"])
            self.assertIsNone(vaes.auto_for(str(own), "sd15", ROWS)[0])
            self.assertEqual(vaes.auto_for(str(bare), "sd15", ROWS)[0], "vae-ft-mse-840000-ema-pruned")

    def test_the_vae_a_model_comes_with_wins(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "animeModel.safetensors"
            fake_safetensors(m, ["first_stage_model.a"])
            (Path(d) / "animeModel.civitai.info").write_text(json.dumps({"vae": "klF8Anime2"}))
            self.assertEqual(vaes.auto_for(str(m), "sd15", ROWS)[0], "klF8Anime2")
            (Path(d) / "animeModel.civitai.info").write_text(json.dumps({"vae": "not-downloaded"}))
            self.assertIsNone(vaes.auto_for(str(m), "sd15", ROWS)[0])

    def test_other_families_untouched(self):
        self.assertEqual(vaes.auto_for(None, "flux", ROWS), (None, ""))

    def test_fp16_fix_names(self):
        self.assertTrue(vaes.is_fp16_fix("sdxl_vae_fp16_fix"))
        self.assertTrue(vaes.is_fp16_fix("sdxl-vae-fp16-fix"))
        self.assertFalse(vaes.is_fp16_fix("sdxl_vae"))


class CivitaiVaeFileTests(unittest.TestCase):
    def test_checkpoint_with_a_vae_file(self):
        version = {"id": 9, "baseModel": "SD 1.5", "files": [
            {"type": "Model", "primary": True, "name": "m.safetensors", "sizeKB": 2000000, "downloadUrl": "https://x/m",
             "metadata": {"format": "SafeTensor"}},
            {"type": "VAE", "name": "m.vae.safetensors", "sizeKB": 320000, "downloadUrl": "https://x/v"}]}
        info = civitai.describe({"id": 1, "type": "Checkpoint", "name": "M"}, version)
        self.assertEqual(info["vae_file"]["name"], "m.vae.safetensors")
        lora = civitai.describe({"id": 1, "type": "LORA", "name": "L"}, version)
        self.assertIsNone(lora["vae_file"])


if __name__ == "__main__":
    unittest.main()
