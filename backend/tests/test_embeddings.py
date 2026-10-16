"""Embeddings (textual inversions): telling them apart from LoRAs, their model type, finding and adding them."""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import civitai  # noqa: E402
import embeddings  # noqa: E402
import loras  # noqa: E402
import paths  # noqa: E402


def write_st(path, shapes, metadata=None):
    header = {k: {"dtype": "F16", "shape": s, "data_offsets": [0, 2]} for k, s in shapes.items()}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\x00\x00")


class ClassifyTests(unittest.TestCase):
    def test_types(self):
        h = lambda **s: {k: {"shape": v} for k, v in s.items()}  # noqa: E731
        self.assertEqual(embeddings.classify(h(emb_params=[8, 768])), {"family": "sd15", "vectors": 8})
        self.assertEqual(embeddings.classify({"string_to_param.*": {"shape": [2, 768]}})["family"], "sd15")
        self.assertEqual(embeddings.classify(h(easynegative=[768]))["vectors"], 1)
        self.assertEqual(embeddings.classify(h(clip_l=[4, 768], clip_g=[4, 1280])), {"family": "sdxl", "vectors": 4})
        self.assertEqual(embeddings.classify(h(emb_params=[3, 1024]))["family"], "sd2")
        self.assertIsNone(embeddings.classify(h(**{"lora_unet_down.lora_up.weight": [4, 768]})))
        self.assertIsNone(embeddings.classify(h(a=[4, 768], b=[4, 768])))  # two unnamed tensors: not one
        self.assertIsNone(embeddings.classify(h(emb_params=[4, 512])))
        self.assertIsNone(embeddings.classify({}))


class FolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (paths.HOME, paths.LORAS_DIR, list(loras.EXTRA_DIRS), list(embeddings.EXTRA_DIRS))
        root = Path(self.tmp.name)
        paths.HOME = root / "home"
        paths.LORAS_DIR = root / "home" / "loras"
        loras.EXTRA_DIRS[:] = []
        embeddings.EXTRA_DIRS[:] = [str(root / "mine")]
        embeddings._cache.clear()
        loras._header_cache.clear()
        self.root = root
        write_st(root / "home" / "embeddings" / "EasyNegative.safetensors", {"emb_params": [8, 768]})
        write_st(root / "mine" / "xl" / "pony_neg.safetensors", {"clip_l": [6, 768], "clip_g": [6, 1280]})
        write_st(root / "mine" / "sd2style.safetensors", {"emb_params": [2, 1024]})
        # an embedding left in the LoRA folder, next to a real LoRA
        write_st(paths.LORAS_DIR / "badhands.safetensors", {"emb_params": [5, 768]})
        write_st(paths.LORAS_DIR / "detail.safetensors", {"lora_unet_down_blocks_0.lora_up.weight": [4, 320]})

    def tearDown(self):
        paths.HOME, paths.LORAS_DIR, le, ee = self.old
        loras.EXTRA_DIRS[:] = le
        embeddings.EXTRA_DIRS[:] = ee
        self.tmp.cleanup()

    def test_found_everywhere_with_their_type(self):
        found = {e["name"]: e for e in embeddings.all_embeddings()}
        self.assertEqual(found["EasyNegative"]["family"], "sd15")
        self.assertEqual(found["EasyNegative"]["source"], "protoart")
        self.assertEqual(found["pony_neg"]["family"], "sdxl")
        self.assertEqual(found["pony_neg"]["source"], "folder")
        self.assertEqual(found["badhands"]["source"], "lora_folder")
        self.assertEqual(found["sd2style"]["family"], "sd2")
        self.assertEqual([e["name"] for e in embeddings.for_family("sdxl")], ["pony_neg"])

    def test_not_listed_as_loras(self):
        names = [l["name"] for l in loras.scan(paths.LORAS_DIR)]
        self.assertEqual(names, ["detail"])

    def test_add_files_copies_only_embeddings(self):
        src = self.root / "downloads"
        write_st(src / "my style.safetensors", {"emb_params": [3, 768]})
        write_st(src / "a_lora.safetensors", {"lora_te_text_model.lora_down.weight": [4, 768]})
        r = embeddings.add_files([str(src / "my style.safetensors"), str(src / "a_lora.safetensors")])
        self.assertEqual(r["added"], ["my_style"])
        self.assertEqual(len(r["skipped"]), 1)
        self.assertTrue((paths.HOME / "embeddings" / "my_style.safetensors").exists())
        again = embeddings.add_files([str(src / "my style.safetensors")])
        self.assertEqual(again["added"], [])  # never overwrites
        self.assertIn("already", again["skipped"][0]["why"])


class CivitaiEmbeddingTests(unittest.TestCase):
    def version(self, base, fname):
        return {"id": 9, "baseModel": base, "files": [{"name": fname, "sizeKB": 25, "type": "Model", "primary": True}]}

    def test_textual_inversion(self):
        info = civitai.describe({"id": 1, "name": "EasyNegative", "type": "TextualInversion"},
                                self.version("SD 1.5", "easynegative.safetensors"))
        self.assertEqual((info["kind"], info["family"], info["ok"]), ("embedding", "sd15", True))
        pt = civitai.describe({"id": 1, "name": "Old", "type": "TextualInversion"}, self.version("Pony", "old.pt"))
        self.assertTrue(pt["ok"])  # .pt embeddings are allowed (weights-only loading)
        self.assertEqual(civitai.safe_filename("old.pt", civitai.allowed_extensions("embedding")), "old.pt")
        sd2 = civitai.describe({"id": 1, "name": "X", "type": "TextualInversion"}, self.version("SD 2.1", "x.safetensors"))
        self.assertFalse(sd2["ok"])
        lora_pt = civitai.describe({"id": 1, "name": "L", "type": "LORA"}, self.version("SD 1.5", "l.pt"))
        self.assertFalse(lora_pt["ok"])  # still safetensors-only for everything else


if __name__ == "__main__":
    unittest.main()
