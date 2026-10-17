"""FLUX.2 [klein] on the fast engine: the model entry, telling its LoRAs from FLUX.1 ones, the live preview,
and the prompt cache that lets small Macs drop the text encoder while drawing."""
import sys
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


if __name__ == "__main__":
    unittest.main()
