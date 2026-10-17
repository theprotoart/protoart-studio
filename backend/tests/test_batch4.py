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
        xl = civitai.describe({"id": 3, "name": "QR XL", "type": "Controlnet"}, _ver("SDXL 1.0"))
        self.assertTrue(xl["ok"], xl["notes"])
        no = civitai.describe({"id": 4, "name": "Flux CN", "type": "Controlnet"}, _ver("Flux.1 D"))
        self.assertFalse(no["ok"])


def _fake_safetensors(path, shapes):
    """A .safetensors file with these weight names/shapes (zero-filled, float16)."""
    import json
    import struct
    head, off = {}, 0
    for k, shp in shapes.items():
        n = 2
        for d in shp:
            n *= d
        head[k] = {"dtype": "F16", "shape": list(shp), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(head).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


class UserControlnetTests(unittest.TestCase):
    def test_user_types_and_map(self):
        import paths
        import controlnet
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            old = paths.CONTROLNET_DIR
            paths.CONTROLNET_DIR = Path(d)
            try:
                sd15 = {"controlnet_cond_embedding.conv_in.weight": (16, 3, 3, 3),
                        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight": (640, 768)}
                _fake_safetensors(os.path.join(d, "qrMonster.safetensors"), sd15)
                xl = {"controlnet_cond_embedding.conv_in.weight": (16, 3, 3, 3), "add_embedding.linear_1.weight": (1280, 2816),
                      "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight": (640, 2048)}
                _fake_safetensors(os.path.join(d, "controlnet852AClone_v10.safetensors"), xl)
                _fake_safetensors(os.path.join(d, "xl_canny_v2.safetensors"), xl)
                _fake_safetensors(os.path.join(d, "controlnetxlUnion.safetensors"), dict(xl, task_embedding=(8, 320)))
                Path(d, "junk.safetensors").write_bytes(b"0" * 10)
                types = {t["id"]: t for t in controlnet.list_types()}
                self.assertIn("user:qrMonster", types)
                self.assertEqual(types["user:qrMonster"]["family"], "sd15")
                self.assertEqual(types["user:controlnet852AClone_v10"]["family"], "sdxl")
                self.assertEqual(controlnet.family_of("user:controlnet852AClone_v10"), "sdxl")
                self.assertEqual(types["user:xl_canny_v2"]["makes_map"], "canny")
                self.assertNotIn("user:controlnetxlUnion", types)  # listed in Models with why, not pickable
                self.assertNotIn("user:junk", types)
                files = {u["name"]: u for u in controlnet.user_files()}
                self.assertIn("Union", files["controlnetxlUnion"]["problem"])
                self.assertTrue(files["junk"]["problem"])
                self.assertEqual(controlnet.family_of("canny"), "sd15")
                canny = controlnet.make_map("user:xl_canny_v2", Image.new("RGB", (40, 30), (200, 10, 10)), 64, 48)
                self.assertEqual(canny.getpixel((5, 5)), (0, 0, 0))  # an edge map, not the picture
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




class XlConfigTests(unittest.TestCase):
    def test_sizes_from_file(self):
        import controlnet

        class T:
            def __init__(self, *shape):
                self.shape = shape

        unet = {"block_out_channels": [320, 640, 1280], "layers_per_block": 2, "cross_attention_dim": 2048,
                "transformer_layers_per_block": [1, 2, 10], "addition_embed_type": "text_time",
                "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]}
        st = {"controlnet_cond_embedding.conv_in.weight": T(16, 3, 3, 3),
              "controlnet_cond_embedding.blocks.1.weight": T(32, 16, 3, 3),
              "controlnet_cond_embedding.blocks.3.weight": T(96, 32, 3, 3),
              "controlnet_cond_embedding.blocks.5.weight": T(256, 96, 3, 3)}
        for b, n in ((1, 2), (2, 10)):
            for a in range(2):
                for t in range(n):
                    st[f"down_blocks.{b}.attentions.{a}.transformer_blocks.{t}.attn1.to_q.weight"] = T(1, 1)
        for t in range(10):
            st[f"mid_block.attentions.0.transformer_blocks.{t}.attn1.to_q.weight"] = T(1, 1)
        cfg = controlnet.xl_config(st, unet)
        self.assertEqual(cfg["transformer_layers_per_block"], (1, 2, 10))
        self.assertEqual(cfg["down_block_types"], ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"))
        self.assertEqual(cfg["mid_block_type"], "UNetMidBlock2DCrossAttn")
        self.assertEqual(cfg["conditioning_embedding_out_channels"], (16, 32, 96, 256))
        self.assertEqual(cfg["conditioning_channels"], 3)
        # "small": fewer layers per block, no middle attention
        small = {k: v for k, v in st.items() if not k.startswith("mid_block")
                 and not any(f"transformer_blocks.{t}." in k for t in range(1, 10))}
        cfg = controlnet.xl_config(small, unet)
        self.assertEqual(cfg["transformer_layers_per_block"], (1, 1, 1))
        self.assertEqual(cfg["mid_block_type"], "UNetMidBlock2D")


if __name__ == "__main__":
    unittest.main()
