"""Training helpers that don't need torch."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trainer  # noqa: E402


class TrainerTests(unittest.TestCase):
    def test_dataset_and_captions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                Image.new("RGB", (64, 64)).save(Path(d) / f"img{i}.png")
            (Path(d) / "img0.txt").write_text("a red dog\n")
            (Path(d) / "notes.md").write_text("x")
            items = trainer.scan_dataset(d)
            self.assertEqual([i["name"] for i in items], ["img0.png", "img1.png", "img2.png"])
            self.assertTrue(items[0]["has_caption"])
            self.assertEqual((items[0]["width"], items[0]["height"]), (64, 64))
            items[1]["caption"] = "a blue cat"
            self.assertEqual(trainer.save_captions(items[1:2]), 1)
            self.assertEqual((Path(d) / "img1.txt").read_text().strip(), "a blue cat")
        with self.assertRaises(trainer.TrainError):
            trainer.scan_dataset("/nonexistent/folder")

    def test_trigger_and_steps(self):
        self.assertEqual(trainer._caption_for({"caption": "a dog"}, "pxl"), "pxl, a dog")
        self.assertEqual(trainer._caption_for({"caption": "pxl, a dog"}, "pxl"), "pxl, a dog")
        self.assertEqual(trainer._caption_for({"caption": ""}, "pxl"), "pxl")
        self.assertEqual(trainer.suggest_steps(2), 400)
        self.assertEqual(trainer.suggest_steps(15), 1500)
        self.assertEqual(trainer.suggest_steps(80), 2000)


class MlxTrainTests(unittest.TestCase):
    def test_progress_and_config(self):
        import mlx_train
        self.assertEqual(mlx_train.parse_progress(" 12%|#   | 120/1000 [05:12<38:10,  2.60s/it]"), (120, 1000, 2.6))
        done, total, sps = mlx_train.parse_progress(" 50%|##  | 5/10 [00:02<00:02,  2.00it/s]")
        self.assertEqual((done, total, sps), (5, 10, 0.5))
        self.assertIsNone(mlx_train.parse_progress("Loading weights"))
        c = mlx_train.build_config(model_path="/m", data_dir="/d", out_dir="/o", steps=1000, n_samples=30, lr=1e-4,
                                   rank=8, res=512, targets="attention", save_every=0, ram_gb=8)
        self.assertEqual(c["model"], "ernie-image-turbo")
        self.assertIsNone(c["quantize"])
        self.assertTrue(c["low_ram"] and c["gradient_checkpointing"])
        self.assertGreaterEqual(c["training_loop"]["num_epochs"] * 30, 1000)
        self.assertEqual({t["rank"] for t in c["lora_layers"]["targets"]}, {8})

    def test_metadata_rewrite_keeps_tensors(self):
        import json
        import struct
        import mlx_train
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.safetensors"
            data = bytes(range(24))
            hdr = json.dumps({"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
            p.write_bytes(struct.pack("<Q", len(hdr)) + hdr + data)
            mlx_train.add_metadata(p, {"beestudio_family": "ernie"})
            raw = p.read_bytes()
            n = struct.unpack("<Q", raw[:8])[0]
            self.assertEqual(json.loads(raw[8:8 + n])["__metadata__"]["beestudio_family"], "ernie")
            self.assertEqual(raw[8 + n:], data)


if __name__ == "__main__":
    unittest.main()


class LocalModelTests(unittest.TestCase):
    def test_inspect_local(self):
        import json as _j
        import struct
        import model_hub

        def write(path, keys):
            h = _j.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(h)) + h + b"\0\0")
        with tempfile.TemporaryDirectory() as d:
            write(Path(d) / "mine.safetensors", ["model.diffusion_model.input_blocks.0.0.weight"])
            write(Path(d) / "xl.safetensors", ["conditioner.embedders.1.model.x"])
            write(Path(d) / "lora.safetensors", ["lora_unet_x.lora_up.weight"])
            self.assertEqual(model_hub.inspect_local(str(Path(d) / "mine.safetensors"))["family"], "sd15")
            self.assertEqual(model_hub.inspect_local(str(Path(d) / "xl.safetensors"))["family"], "sdxl")
            with self.assertRaises(ValueError):
                model_hub.inspect_local(str(Path(d) / "lora.safetensors"))
            folder = Path(d) / "ernie_full"
            folder.mkdir()
            (folder / "model_index.json").write_text(_j.dumps({"_class_name": "ErnieImagePipeline"}))
            info = model_hub.inspect_local(str(folder))
            self.assertEqual((info["family"], info["kind"]), ("ernie", "folder"))
            spec = model_hub.local_to_spec(info)
            self.assertEqual(spec["local_dir"], str(folder))
