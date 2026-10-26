import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # test_tdict's writer
import lora_extract as L


class NameTests(unittest.TestCase):
    def test_unet_names_follow_diffusers_blocks(self):
        pre = "model.diffusion_model."
        self.assertEqual(L.unet_kohya_name(pre + "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"),
                         "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q")
        self.assertEqual(L.unet_kohya_name(pre + "input_blocks.8.1.proj_in.weight"), "lora_unet_down_blocks_2_attentions_1_proj_in")
        self.assertEqual(L.unet_kohya_name(pre + "output_blocks.5.1.proj_out.weight"), "lora_unet_up_blocks_1_attentions_2_proj_out")
        self.assertEqual(L.unet_kohya_name(pre + "middle_block.1.transformer_blocks.0.ff.net.2.weight"),
                         "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_2")
        self.assertIsNone(L.unet_kohya_name(pre + "input_blocks.1.0.in_layers.2.weight"))

    def test_text_encoder_names(self):
        self.assertEqual(L.te_kohya_name("cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight"),
                         "lora_te_text_model_encoder_layers_0_mlp_fc1")
        self.assertEqual(L.te_kohya_name("conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight"),
                         "lora_te1_text_model_encoder_layers_2_self_attn_q_proj")


class ExtractTests(unittest.TestCase):
    def test_extracts_the_difference(self):
        try:
            import torch
            from safetensors.torch import save_file, load_file
        except Exception:
            self.skipTest("torch / safetensors not installed")
        torch.manual_seed(0)
        k = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        k2 = "model.diffusion_model.input_blocks.1.1.proj_in.weight"
        base = {k: torch.randn(64, 64), k2: torch.randn(64, 64, 1, 1)}
        delta = torch.randn(64, 4) @ torch.randn(4, 64) * 0.1  # rank 4 change
        tuned = {k: base[k] + delta, k2: base[k2].clone()}
        with tempfile.TemporaryDirectory() as d:
            save_file(base, os.path.join(d, "base.safetensors"))
            save_file(tuned, os.path.join(d, "tuned.safetensors"))
            old = L.paths.LORAS_DIR
            L.paths.LORAS_DIR = Path(d) / "loras"
            try:
                r = L.extract(os.path.join(d, "tuned.safetensors"), os.path.join(d, "base.safetensors"), rank=8)
                sd = load_file(r["path"])
            finally:
                L.paths.LORAS_DIR = old
        n = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
        rebuilt = sd[n + ".lora_up.weight"].float() @ sd[n + ".lora_down.weight"].float()
        self.assertLess((rebuilt - delta).abs().max().item(), 0.02)
        self.assertEqual(r["layers"], 1)
        self.assertEqual(r["unchanged"], 1)

    def test_straight_from_a_diffusionbee_tdict(self):
        try:
            import numpy as np
            import torch
            from safetensors.torch import save_file, load_file
        except Exception:
            self.skipTest("torch / safetensors not installed")
        from test_tdict import write_tdict
        torch.manual_seed(1)
        k = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        base = {k: torch.randn(64, 64)}
        delta = torch.randn(64, 4) @ torch.randn(4, 64) * 0.1
        with tempfile.TemporaryDirectory() as d:
            save_file(base, os.path.join(d, "base.safetensors"))
            tuned = os.path.join(d, "myModel_myModel.tdict")
            write_tdict(tuned, {k: (base[k] + delta).numpy().astype(np.float16)}, 12)  # 12 = an SD 1.5 model
            src = open(tuned, "rb").read()
            old = L.paths.LORAS_DIR
            L.paths.LORAS_DIR = Path(d) / "loras"
            try:
                r = L.extract(tuned, os.path.join(d, "base.safetensors"), rank=8)
                sd = load_file(r["path"])
            finally:
                L.paths.LORAS_DIR = old
            self.assertEqual(open(tuned, "rb").read(), src)  # the .tdict isn't changed
        self.assertEqual(r["name"], "myModel-lora-r8")
        n = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
        rebuilt = sd[n + ".lora_up.weight"].float() @ sd[n + ".lora_down.weight"].float()
        self.assertLess((rebuilt - delta).abs().max().item(), 0.03)

    def test_resnet_names(self):
        pre = "model.diffusion_model."
        self.assertEqual(L.res_kohya_name(pre + "input_blocks.1.0.in_layers.2.weight"), "lora_unet_down_blocks_0_resnets_0_conv1")
        self.assertEqual(L.res_kohya_name(pre + "input_blocks.5.0.skip_connection.weight"), "lora_unet_down_blocks_1_resnets_1_conv_shortcut")
        self.assertEqual(L.res_kohya_name(pre + "middle_block.2.out_layers.3.weight"), "lora_unet_mid_block_resnets_1_conv2")
        self.assertEqual(L.res_kohya_name(pre + "output_blocks.7.0.emb_layers.1.weight"), "lora_unet_up_blocks_2_resnets_1_time_emb_proj")
        self.assertIsNone(L.res_kohya_name(pre + "input_blocks.3.0.op.weight"))

    def test_resnet_convs_are_kept(self):
        try:
            import torch
            from safetensors.torch import save_file, load_file
        except Exception:
            self.skipTest("torch / safetensors not installed")
        torch.manual_seed(3)
        k = "model.diffusion_model.input_blocks.1.0.in_layers.2.weight"
        ka = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        base = {k: torch.randn(32, 16, 3, 3), ka: torch.randn(64, 64)}
        delta = (torch.randn(32, 4) @ torch.randn(4, 16 * 9) * 0.1).reshape(32, 16, 3, 3)  # a rank-4 change
        tuned = {k: base[k] + delta, ka: base[ka] + torch.randn(64, 2) @ torch.randn(2, 64) * 0.1}
        with tempfile.TemporaryDirectory() as d:
            save_file(base, os.path.join(d, "base.safetensors"))
            save_file(tuned, os.path.join(d, "tuned.safetensors"))
            old = L.paths.LORAS_DIR
            L.paths.LORAS_DIR = Path(d) / "loras"
            try:
                r = L.extract(os.path.join(d, "tuned.safetensors"), os.path.join(d, "base.safetensors"), rank=8)
                sd = load_file(r["path"])
            finally:
                L.paths.LORAS_DIR = old
        n = "lora_unet_down_blocks_0_resnets_0_conv1"
        down, up = sd[n + ".lora_down.weight"].float(), sd[n + ".lora_up.weight"].float()
        self.assertEqual(tuple(down.shape[1:]), (16, 3, 3))
        self.assertEqual(tuple(up.shape[2:]), (1, 1))
        rebuilt = (up[:, :, 0, 0] @ down.reshape(down.shape[0], -1)).reshape(32, 16, 3, 3)
        self.assertLess((rebuilt - delta).abs().max().item(), 0.03)
        self.assertGreater(r["covered_pct"], 99)

    def test_a_tdict_lora_is_refused_kindly(self):
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not installed")
        from test_tdict import write_tdict
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "style.tdict")
            write_tdict(p, {"lora_unet_x_lora_up": np.zeros((4, 4), np.float32)}, 0)
            with self.assertRaises(ValueError) as e:
                L.open_weights(p)
            self.assertIn("Convert", str(e.exception))
            with self.assertRaises(ValueError):
                L.open_weights(os.path.join(d, "missing.safetensors"))


if __name__ == "__main__":
    unittest.main()
