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


class FramingTests(unittest.TestCase):
    """Train tab: crop each picture to a square (centre/top/bottom/left/right) or keep it whole (pad)."""

    def test_square_fits(self):
        import trainer
        from PIL import Image
        img = Image.new("RGB", (300, 100), (255, 0, 0))
        img.paste((0, 0, 255), (0, 0, 100, 100))  # blue on the left third
        self.assertEqual(trainer.square(img, 64).size, (64, 64))
        self.assertEqual(trainer.square(img, 64, "left").getpixel((32, 32)), (0, 0, 255))
        self.assertEqual(trainer.square(img, 64, "right").getpixel((32, 32)), (255, 0, 0))
        padded = trainer.square(img, 60, "pad")
        self.assertEqual(padded.size, (60, 60))
        self.assertEqual(padded.getpixel((5, 30))[2], 255)  # whole picture kept: blue at the left edge

    def test_framing_saved_and_copies(self):
        import tempfile
        from pathlib import Path
        import trainer
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "set"
            d.mkdir()
            for n in ("a", "b", "c"):
                Image.new("RGB", (200, 120), (10, 20, 30)).save(d / f"{n}.png")
            (d / "a.txt").write_text("a cat\n")
            self.assertEqual(trainer.save_framing(d, {"a.png": "top", "b.png": "center", "zzz.png": "pad"}), 1)
            items = {i["name"]: i["fit"] for i in trainer.scan_dataset(d)}
            self.assertEqual(items, {"a.png": "top", "b.png": "center", "c.png": "center"})
            r = trainer.save_square_copies(d, 64)
            out = Path(r["folder"])
            self.assertEqual(r["count"], 3)
            self.assertEqual(Image.open(out / "a.png").size, (64, 64))
            self.assertEqual((out / "a.txt").read_text().strip(), "a cat")
            self.assertEqual(trainer.save_framing(d, {}), 0)
            self.assertFalse((d / trainer.FRAMING_FILE).exists())
            # left out: listed for the Train tab, never trained on or copied
            trainer.save_framing(d, {"b.png": "skip", "c.png": "box:0.1,0,0.5"})
            self.assertEqual([i["name"] for i in trainer.scan_dataset(d)], ["a.png", "c.png"])
            self.assertEqual(len(trainer.scan_dataset(d, include_skipped=True)), 3)
            self.assertEqual(trainer.save_square_copies(d, 64)["count"], 2)

    def test_box_crop(self):
        import trainer
        from PIL import Image
        img = Image.new("RGB", (400, 200), (255, 0, 0))
        img.paste((0, 255, 0), (300, 100, 400, 200))  # green bottom-right corner, 100 px
        out = trainer.square(img, 50, "box:0.75,0.5,0.5")  # the 100 px square at (300, 100)
        self.assertEqual(out.size, (50, 50))
        self.assertEqual(out.getpixel((25, 25)), (0, 255, 0))
        self.assertIsNone(trainer.parse_box("box:a,b,c"))
        self.assertTrue(trainer.valid_fit("skip"))
        self.assertFalse(trainer.valid_fit("box:0,0,5"))
