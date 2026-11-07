"""📦 Moving the app's model files to another drive: files move, references follow, the setting sticks, and back."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import json, os, sys, threading
from pathlib import Path
import bee_backend as b, paths, loras
home, ext = Path(os.environ["PROTOART_HOME"]), Path(sys.argv[1])
(paths.LORAS_DIR / "flux2").mkdir(parents=True, exist_ok=True)
(paths.LORAS_DIR / "flux2" / "a.safetensors").write_bytes(b"x" * 1000)
(paths.MODELS_DIR / "m.safetensors").write_bytes(b"y" * 2000)
loras.FOLDER_FAMILY[str(paths.LORAS_DIR / "flux2")] = "flux2"
class E:
    cancel_event = threading.Event()
    def emit(self, m): pass
r = b.store_move({"dest": str(ext)}, E())
info = b.store_info()
fam_keys = list(loras.FOLDER_FAMILY)
saved = b.library.get_setting("model_store")
there = (ext / "Protoart models" / "loras" / "flux2" / "a.safetensors").exists()
back = b.store_move({"dest": ""}, E())
sys.__stdout__.write("RESULT" + json.dumps({"r": r, "info_root": info["root"], "fam": fam_keys, "saved": saved,
                  "lora_there": there,
                  "back_lora": (home / "loras" / "flux2" / "a.safetensors").exists(),
                  "back_model": (home / "models" / "m.safetensors").exists(),
                  "saved_after": b.library.get_setting("model_store"), "loras_dir": str(paths.LORAS_DIR)}) + "\n")
'''


class StoreMove(unittest.TestCase):
    def test_move_and_back(self):
        home, ext = tempfile.mkdtemp(), tempfile.mkdtemp()
        p = subprocess.run([sys.executable, "-c", SCRIPT, ext], cwd=str(HERE), capture_output=True, text=True,
                           env=dict(os.environ, PROTOART_HOME=home), timeout=120)
        line = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith("RESULT")]
        self.assertTrue(line, p.stderr[-2000:])
        d = json.loads(line[-1][6:])
        self.assertTrue(d["lora_there"])
        self.assertEqual(d["r"]["failed"], [])
        self.assertEqual(d["info_root"], str(Path(ext) / "Protoart models"))
        self.assertEqual(d["saved"], str(Path(ext) / "Protoart models"))
        self.assertTrue(any(k.startswith(str(Path(ext) / "Protoart models")) for k in d["fam"]))  # LoRA type followed
        self.assertTrue(d["back_lora"] and d["back_model"])
        self.assertIsNone(d["saved_after"])
        self.assertEqual(d["loras_dir"], str(Path(home) / "loras"))


if __name__ == "__main__":
    unittest.main()
