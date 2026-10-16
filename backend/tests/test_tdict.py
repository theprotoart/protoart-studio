"""Tests for reading DiffusionBee .tdict files (no torch needed)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tdict_reader  # noqa: E402
import samplers  # noqa: E402


def write_tdict(path, tensors, ctdict):
    """Minimal writer producing the same layout DiffusionBee writes."""
    keys_info = {}
    body = bytearray()
    base = 64 + (64 + 160) + 32  # file head + extra-head block, padded to 64 -> 320
    json_area = 320 + 64          # weights JSON data starts after its block header
    json_size = 200_000
    meta_start = json_area + json_size + 64
    pos = meta_start + 64 + 64    # metadata area (64 bytes), then tensors
    for k, arr in tensors.items():
        pos += (-pos) % 64
        pos += 64  # block header
        keys_info[k] = {"start": pos, "end": pos + arr.nbytes, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        pos += arr.nbytes
    wj = json.dumps(keys_info).encode()
    assert len(wj) < json_size
    extra = [4346464, 2, 2, ctdict, json_area, json_area + len(wj), meta_start, meta_start + 2] + [0] * 12
    with open(path, "wb") as f:
        f.write(np.array([8589935844, 0, 0, 0, 0, 0, 0, 0], dtype="<u8").tobytes())
        f.write(np.array([8030895855, 160, 128, 0, 2, 0, 0, 0], dtype="<u8").tobytes())
        f.write(np.array(extra, dtype="<u8").tobytes())
        f.seek(json_area)
        f.write(wj)
        f.seek(meta_start)
        f.write(b"{}")
        for k, arr in tensors.items():
            f.seek(keys_info[k]["start"])
            f.write(arr.tobytes())


class TDictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lora_to_kohya(self):
        up = np.random.rand(320, 4, 1, 1).astype(np.float32)
        down = np.random.rand(4, 320, 1, 1).astype(np.float32)
        te_up = np.random.rand(3072, 4).astype(np.float32)
        te_down = np.random.rand(4, 768).astype(np.float32)
        p = self.dir / "detail_detail.tdict"
        write_tdict(p, {
            "model.diffusion_model.input_blocks.1.1.proj_in.weight_lora_up": up,
            "model.diffusion_model.input_blocks.1.1.proj_in.weight_lora_down": down,
            "model.diffusion_model.input_blocks.1.1.proj_in.weight_lora_scale": np.array(0.5, dtype=np.float32),
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight_lora_up": te_up,
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight_lora_down": te_down,
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight_lora_scale": np.array(1.0, dtype=np.float32),
        }, ctdict=69)
        info = tdict_reader.describe(p)
        self.assertEqual((info["kind"], info["family"], info["name"], info["supported"]), ("lora", "sd15", "detail", True))
        sd = tdict_reader.TDict(p).lora_arrays()
        np.testing.assert_array_equal(sd["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"], up)
        self.assertAlmostEqual(float(sd["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"]), 2.0)  # 0.5 * rank 4
        self.assertIn("lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", sd)
        self.assertAlmostEqual(float(sd["lora_te_text_model_encoder_layers_0_mlp_fc1.alpha"]), 4.0)

    def test_checkpoint_keys_skip_helpers(self):
        z = np.zeros(4, dtype=np.float16)
        p = self.dir / "mymodel_mymodel.tdict"
        write_tdict(p, {
            "model.diffusion_model.time_embed.0.weight": z,
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight._split_1": z,
            "model.diffusion_model.input_blocks.1.1.norm.bias_by_weight": z,
            "first_stage_model.decoder.mid.attn_1.q.weight": z,
            "cond_stage_model.transformer.text_model.final_layer_norm.weight": z,
            "zeros_320": z, "temb_coefficients_fp16": z, "causal_mask": z, "alphas_cumprod": z,
        }, ctdict=1012)
        info = tdict_reader.describe(p)
        self.assertEqual((info["kind"], info["family"], info["supported"]), ("checkpoint", "sd15", True))
        keys = sorted(tdict_reader.TDict(p).checkpoint_keys())
        self.assertEqual(keys, ["cond_stage_model.transformer.text_model.final_layer_norm.weight",
                                "first_stage_model.decoder.mid.attn_1.q.weight",
                                "model.diffusion_model.time_embed.0.weight"])

    def test_embedding_and_unsupported(self):
        p = self.dir / "easyneg_easyneg.tdict"
        write_tdict(p, {"emb_params": np.zeros((8, 768), dtype=np.float16)}, ctdict=87)
        info = tdict_reader.describe(p)
        self.assertEqual((info["kind"], info["family"], info["supported"]), ("embedding", "sd15", True))
        c = self.dir / "cn_cn.tdict"
        write_tdict(c, {"control_model.x": np.zeros(2, dtype=np.float16)}, ctdict=1014)
        self.assertFalse(tdict_reader.describe(c)["supported"])
        found = {i["name"]: i for i in tdict_reader.scan([self.dir])}
        self.assertEqual(set(found), {"easyneg", "cn"})

    def test_bad_file(self):
        p = self.dir / "junk.tdict"
        p.write_bytes(os.urandom(400))
        with self.assertRaises(tdict_reader.TDictError):
            tdict_reader.read_header(p)
        self.assertFalse(tdict_reader.scan([self.dir])[0]["supported"])

    def test_unet_names(self):
        f = tdict_reader.unet_ldm_to_diffusers
        self.assertEqual(f("input_blocks.4.1.transformer_blocks.0.attn1.to_q"), "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q")
        self.assertEqual(f("input_blocks.8.1.proj_in"), "down_blocks.2.attentions.1.proj_in")
        self.assertEqual(f("middle_block.1.proj_out"), "mid_block.attentions.0.proj_out")
        self.assertEqual(f("output_blocks.11.1.proj_out"), "up_blocks.3.attentions.2.proj_out")
        self.assertEqual(f("output_blocks.3.1.transformer_blocks.0.ff.net.2"), "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2")
        self.assertEqual(f("input_blocks.3.0.op"), "down_blocks.0.downsamplers.0.conv")
        self.assertIsNone(tdict_reader.kohya_name("model.diffusion_model.input_blocks.1.1.ff.net.0.proj.weight._split_1"))

    def test_clean_name(self):
        self.assertEqual(tdict_reader.clean_name("/x/more_details_more_details.tdict"), "more_details")
        self.assertEqual(tdict_reader.clean_name("/x/Default_SDB_0.1_SDB_0.1.tdict"), "Default_SDB_0.1_SDB_0.1")


class TriggerTests(unittest.TestCase):
    def test_meta_and_sidecars(self):
        import loras
        meta = {"ss_tag_frequency": json.dumps({"10_x": {"pncl style": 40, "1girl": 50, "sketch": 30}}),
                "modelspec.trigger_phrase": "pncl style, graphite"}
        words, sugg = loras.triggers_from_meta(meta)
        self.assertEqual(words, ["pncl style", "graphite"])
        self.assertEqual(sugg[:2], ["pncl style", "sketch"])  # generic tags like 1girl skipped
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "pencil.safetensors"
            f.write_bytes(b"x")
            (Path(d) / "pencil.civitai.info").write_text(json.dumps({"trainedWords": ["pencil drawing", "lineart"]}))
            self.assertEqual(loras.triggers_from_sidecars(f), ["pencil drawing", "lineart"])


class RecommendedStepsTests(unittest.TestCase):
    def test_every_sampler_has_steps(self):
        for s in samplers.list_for("sd15"):
            self.assertTrue(s["steps"] and s["steps"][1] <= s["steps"][0] <= s["steps"][2], s)

    def test_flow_uses_model_default(self):
        self.assertEqual(samplers.recommended_steps("ernie", "flow_euler", 8)[0], 8)
        self.assertEqual(samplers.recommended_steps("sd15", "lcm")[0], 6)


if __name__ == "__main__":
    unittest.main()


class PreviewTests(unittest.TestCase):
    def test_previews(self):
        import preview
        for fam, shape in (("sd15", (4, 64, 64)), ("sdxl", (4, 128, 96)), ("ernie", (128, 32, 48))):
            rgb = preview.latent_to_rgb(np.random.randn(*shape).astype(np.float32) * 10, fam)
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertEqual(rgb.shape[2], 3)
        self.assertTrue(preview.latent_to_data_url(np.zeros((4, 8, 8)), "sd15").startswith("data:image/jpeg;base64,"))


class ConvertTests(unittest.TestCase):
    def test_convert_lora_and_embedding(self):
        import json as _j
        import struct as _s
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "style_style.tdict"
            write_tdict(p, {
                "model.diffusion_model.input_blocks.1.1.proj_in.weight_lora_up": np.ones((320, 4, 1, 1), np.float32),
                "model.diffusion_model.input_blocks.1.1.proj_in.weight_lora_down": np.ones((4, 320, 1, 1), np.float32),
                "model.diffusion_model.input_blocks.1.1.proj_in.weight_lora_scale": np.array(1.0, np.float32),
            }, ctdict=69)
            out = tdict_reader.convert_to_safetensors(p, Path(d) / "out")
            b = Path(out).read_bytes()
            n = _s.unpack("<Q", b[:8])[0]
            h = _j.loads(b[8:8 + n])
            self.assertIn("lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight", h)
            self.assertEqual(h["__metadata__"]["protoart_kind"], "lora")
            e = Path(d) / "neg_neg.tdict"
            write_tdict(e, {"emb_params": np.zeros((4, 768), np.float16)}, ctdict=87)
            self.assertTrue(tdict_reader.convert_to_safetensors(e, Path(d) / "out").endswith("neg.safetensors"))
