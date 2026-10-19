"""FLUX.2 [klein] on the fast engine: the model entry, telling its LoRAs from FLUX.1 ones, the live preview,
and the prompt cache that lets small Macs drop the text encoder while drawing."""
import sys
import os
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import civitai  # noqa: E402
import loras  # noqa: E402
import preview  # noqa: E402
import samplers  # noqa: E402


def _header(keys, meta=None):
    h = {k: {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]} for k in keys}
    if meta is not None:
        h["__metadata__"] = meta
    return h


def _blocks(n_double, n_single, style="kohya"):
    if style == "kohya":
        return ([f"lora_unet_double_blocks_{i}_img_attn_qkv.lora_down.weight" for i in range(n_double)]
                + [f"lora_unet_single_blocks_{i}_linear1.lora_down.weight" for i in range(n_single)])
    return ([f"transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight" for i in range(n_double)]
            + [f"transformer.single_transformer_blocks.{i}.attn.to_q.lora_A.weight" for i in range(n_single)])


class ModelEntryTests(unittest.TestCase):
    def test_entry(self):
        try:
            import engine
        except Exception as e:  # torch etc. not installed here
            self.skipTest(f"engine not importable: {e}")
        spec = engine.MODELS["flux2_klein_mlx_q4"]
        self.assertEqual(spec["family"], "flux2")
        self.assertEqual(spec["engine"], "mlx")
        self.assertEqual(spec["mlx_base"], "flux2-klein-4b")
        self.assertEqual(len(spec["mlx_revision"]), 40)  # pinned to one commit
        self.assertLessEqual(spec["min_ram_gb"], 8)
        self.assertEqual(spec["license"], "Apache-2.0")


class LoraFamilyTests(unittest.TestCase):
    def test_metadata(self):
        self.assertEqual(loras.detect_family(_header(["a"], {"ss_base_model_version": "flux2_klein_4b"})), "flux2")
        self.assertEqual(loras.detect_family(_header(["a"], {"modelspec.architecture": "Flux.2-Klein/lora"})), "flux2")
        self.assertEqual(loras.detect_family(_header(["a"], {"ss_base_model_version": "flux1"})), "flux")

    def test_block_count(self):
        for style in ("kohya", "diffusers"):
            self.assertEqual(loras.detect_family(_header(_blocks(5, 20, style))), "flux2", style)
            self.assertEqual(loras.detect_family(_header(_blocks(19, 38, style))), "flux", style)
        # only a few early blocks: can't tell, stays FLUX.1
        self.assertEqual(loras.detect_family(_header(_blocks(2, 3))), "flux")

    def test_civitai_base(self):
        self.assertEqual(civitai.family_of("Flux.2 Klein 4B")[0], "flux2")
        self.assertEqual(civitai.family_of("Flux.2 D")[0], "flux2")
        self.assertEqual(civitai.family_of("Flux.1 D")[0], "flux")


class PreviewAndSamplerTests(unittest.TestCase):
    def test_preview(self):
        tokens = np.random.default_rng(0).standard_normal((48 * 32, 128)).astype(np.float32)
        lat = preview.unpack_flux2_tokens(tokens, 768, 512)
        self.assertEqual(lat.shape, (128, 48, 32))
        rgb = preview.latent_to_rgb(lat, "flux2")
        self.assertEqual(rgb.shape, (96, 64, 3))

    def test_samplers(self):
        self.assertEqual(list(samplers.table_for("flux2")), ["flow_euler"])
        self.assertEqual(samplers.recommended_steps("flux2", "flow_euler", 4)[0], 4)


class PromptCacheTests(unittest.TestCase):
    def test_cache_survives_dropped_encoder(self):
        import mlx_worker
        calls = []

        class FakeKlein:
            def _encode_prompt_pair(self, *, prompt, negative_prompt, guidance):
                calls.append(prompt)
                return (f"emb:{prompt}", None, None, None)

        w = mlx_worker.Worker()
        m = FakeKlein()
        w._cache_flux2_prompts(m)
        self.assertEqual(m._encode_prompt_pair(prompt="a cat", negative_prompt=" ", guidance=1.0)[0], "emb:a cat")
        m._encode_prompt_pair(prompt="a cat", negative_prompt=" ", guidance=1.0)
        self.assertEqual(calls, ["a cat"])           # encoded once
        self.assertIn(("a cat", False), w.f2_cache)  # the key generate() looks for
        self.assertEqual(mlx_worker._family("flux2-klein-4b"), "flux2")
        self.assertEqual(mlx_worker._family("schnell"), "flux")
        self.assertEqual(mlx_worker._family("ernie-image-turbo"), "ernie")




class LoraBakeTests(unittest.TestCase):
    def test_small_macs_keep_loras_unfolded(self):
        import mlx_worker
        old = os.environ.pop("PROTOART_BAKE_LORA", None)
        try:
            real = mlx_worker.total_ram_gb
            mlx_worker.total_ram_gb = lambda: 8.0
            self.assertFalse(mlx_worker.bake_loras())
            mlx_worker.total_ram_gb = lambda: 64.0
            self.assertTrue(mlx_worker.bake_loras())
            os.environ["PROTOART_BAKE_LORA"] = "1"
            mlx_worker.total_ram_gb = lambda: 8.0
            self.assertTrue(mlx_worker.bake_loras())
        finally:
            mlx_worker.total_ram_gb = real
            os.environ.pop("PROTOART_BAKE_LORA", None)
            if old is not None:
                os.environ["PROTOART_BAKE_LORA"] = old

    def test_old_mflux_without_the_option(self):
        import mlx_worker

        class Old:
            def __init__(self, model_path=None, lora_paths=None):
                self.p = model_path

        self.assertEqual(mlx_worker._make(Old, model_path="x", lora_paths=None, bake_lora=False).p, "x")


if __name__ == "__main__":
    unittest.main()


class OutpaintCanvasTests(unittest.TestCase):
    def test_green_border_where_the_new_area_goes(self):
        from PIL import Image
        from engine import Engine
        img = Image.new("RGB", (100, 60), (10, 20, 30))
        canvas, box = Engine.green_canvas(img, left=20, right=0, top=5, bottom=15)
        self.assertEqual(canvas.size, (120, 80))
        self.assertEqual(box, (20, 5, 120, 65))
        self.assertEqual(canvas.getpixel((0, 0)), (0, 255, 0))
        self.assertEqual(canvas.getpixel((119, 79)), (0, 255, 0))
        self.assertEqual(canvas.getpixel((20, 5)), (10, 20, 30))
        self.assertEqual(canvas.getpixel((119, 64)), (10, 20, 30))

    def test_lora_pinned(self):
        from engine import Engine
        L = Engine.OUTPAINT_LORA
        self.assertEqual(len(L["revision"]), 40)
        self.assertEqual(L["prompt"], "Fill the green spaces according to the image")


class OutpaintEnhanceStyleTests(unittest.TestCase):
    def test_outpaint_style(self):
        from postprocess.prompt_enhance import style_for, SYSTEM, EXAMPLES
        self.assertEqual(style_for("outpaint"), "outpaint")
        self.assertIn("outpaint", SYSTEM)
        self.assertIn("outpaint", EXAMPLES)


class OutpaintAlignTests(unittest.TestCase):
    def test_lines_up_a_shifted_result(self):
        import numpy as np
        from PIL import Image, ImageFilter
        from engine import Engine
        rng = np.random.default_rng(1)
        orig = Image.fromarray((rng.random((200, 160, 3)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
        box = (50, 20, 210, 220)
        for sx, sy in [(10, 0), (0, -7), (-12, 5)]:
            res = Image.new("RGB", (260, 240), (90, 90, 90))
            res.paste(orig, (50 + sx, 20 + sy))
            aligned, shift, diff = Engine.align_to_original(res, orig, box)
            self.assertEqual(shift, (-sx, -sy))
            self.assertLess(diff, 0.05)

    def test_redrawn_picture_is_noticed(self):
        import numpy as np
        from PIL import Image, ImageFilter
        from engine import Engine
        rng = np.random.default_rng(2)
        orig = Image.fromarray((rng.random((200, 160, 3)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
        other = Image.fromarray((rng.random((240, 260, 3)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
        _, _, diff = Engine.align_to_original(other, orig, (50, 20, 210, 220))
        self.assertGreater(diff, 0.35)
