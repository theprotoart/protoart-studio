import json
import os
import tempfile
import unittest

import libraries


def touch(p, size=10):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"\0" * size)


class LibraryDetectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        out = {}
        for root, _, files in os.walk(self.h):
            for f in files:
                p = os.path.join(root, f)
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
        return out

    def test_stability_matrix_from_its_settings(self):
        lib = os.path.join(self.h, "Drive", "SM")
        touch(os.path.join(lib, "Models", "StableDiffusion", "dream.safetensors"))
        touch(os.path.join(lib, "Models", "Lora", "a.safetensors"))
        touch(os.path.join(lib, "Models", "Lora", "notes.txt"))
        cfg = os.path.join(self.h, "Library", "Application Support", "StabilityMatrix", "library.json")
        os.makedirs(os.path.dirname(cfg))
        with open(cfg, "w") as f:
            json.dump({"LibraryPath": lib}, f)
        before = self.snapshot()
        found = libraries.detect(self.h)
        self.assertEqual(self.snapshot(), before)  # read-only
        sm = [f for f in found if f["layout"] == "stability_matrix"]
        self.assertEqual(len(sm), 1)
        self.assertEqual(sm[0]["counts"], {"checkpoints": 1, "loras": 1})
        self.assertIn("1 LoRAs", libraries.summary_line(sm[0]))

    def test_comfyui_and_a1111_default_places(self):
        touch(os.path.join(self.h, "ComfyUI", "models", "checkpoints", "x.safetensors"))
        touch(os.path.join(self.h, "ComfyUI", "main.py"))
        touch(os.path.join(self.h, "stable-diffusion-webui", "models", "Stable-diffusion", "y.ckpt"))
        touch(os.path.join(self.h, "stable-diffusion-webui", "embeddings", "e.pt"))
        layouts = sorted(f["layout"] for f in libraries.detect(self.h))
        self.assertEqual(layouts, ["a1111", "comfyui"])

    def test_same_folder_listed_once(self):
        touch(os.path.join(self.h, "StabilityMatrix", "Data", "Models", "Lora", "a.safetensors"))
        os.symlink(os.path.join(self.h, "StabilityMatrix"), os.path.join(self.h, "Documents_SM"))
        self.assertEqual(len(libraries.detect(self.h)), 1)

    def test_pick_models_folder_itself(self):
        lib = os.path.join(self.h, "X")
        touch(os.path.join(lib, "Models", "StableDiffusion", "a.safetensors"))
        touch(os.path.join(lib, "Packages", "keep"))
        info = libraries.inspect_any(os.path.join(lib, "Models"))
        self.assertEqual(info["layout"], "stability_matrix")
        self.assertEqual(info["root"], os.path.realpath(lib))

    def test_plain_folder_is_not_a_library(self):
        touch(os.path.join(self.h, "plain", "a.safetensors"))
        with self.assertRaises(ValueError):
            libraries.inspect_any(os.path.join(self.h, "plain"))

    def test_nothing_found(self):
        self.assertEqual(libraries.detect(self.h), [])


if __name__ == "__main__":
    unittest.main()
