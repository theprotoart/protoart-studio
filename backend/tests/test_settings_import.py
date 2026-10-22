import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings_import as si

A1111 = ("masterpiece, a cat in a hat <lora:pencil:0.6>\nNegative prompt: blurry, lowres\n"
         "Steps: 25, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7.5, Seed: 1234, Size: 512x768, "
         "Model hash: abc, Model: dreamshaper_8, Clip skip: 2")
COMFY = {"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20, "cfg": 6.5, "sampler_name": "euler",
                                                   "scheduler": "normal", "denoise": 1, "positive": ["6", 0], "negative": ["7", 0],
                                                   "model": ["10", 0], "latent_image": ["5", 0]}},
         "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
         "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "style/ink.safetensors", "strength_model": 0.7}},
         "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
         "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox in snow", "clip": ["4", 1]}},
         "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "ugly", "clip": ["4", 1]}}}


class SettingsImportTests(unittest.TestCase):
    def test_a1111(self):
        r = si.parse_a1111(A1111)
        self.assertEqual(r["prompt"], "masterpiece, a cat in a hat")
        self.assertEqual(r["negative_prompt"], "blurry, lowres")
        self.assertEqual((r["steps"], r["cfg"], r["seed"], r["width"], r["height"]), (25, 7.5, 1234, 512, 768))
        self.assertEqual(r["model_name"], "dreamshaper_8")
        self.assertEqual(r["loras"], [{"name": "pencil", "weight": 0.6}])
        self.assertEqual(r["clip_skip"], 2)

    def test_comfy_api(self):
        r = si.parse_comfy(COMFY)
        self.assertEqual((r["prompt"], r["negative_prompt"], r["seed"], r["steps"], r["width"]), ("a fox in snow", "ugly", 42, 20, 832))
        self.assertEqual(r["loras"], [{"name": "ink", "weight": 0.7}])
        self.assertEqual(r["model_name"], "sdxl.safetensors")

    def test_png_with_a1111_text_and_json_file(self):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        d = Path(tempfile.mkdtemp())
        meta = PngInfo()
        meta.add_text("parameters", A1111)
        Image.new("RGB", (8, 8)).save(d / "a.png", pnginfo=meta)
        self.assertEqual(si.read(d / "a.png")["seed"], 1234)
        (d / "w.json").write_text(json.dumps(COMFY))
        self.assertEqual(si.read(d / "w.json")["prompt"], "a fox in snow")
        (d / "p.json").write_text(json.dumps({"app": "Protoart Studio", "kind": "settings",
                                              "settings": {"prompt": "a boat", "steps": 30, "cfg": 5, "seed": 9}}))
        self.assertEqual(si.read(d / "p.json")["steps"], 30)
        Image.new("RGB", (8, 8)).save(d / "plain.png")
        with self.assertRaises(ValueError):
            si.read(d / "plain.png")


if __name__ == "__main__":
    unittest.main()
