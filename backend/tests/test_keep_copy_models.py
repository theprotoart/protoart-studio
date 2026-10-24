import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine
import local_copy
import model_hub
import paths


def fake_sd15(path):
    header = json.dumps({"model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [320, 4, 3, 3],
                                                                            "data_offsets": [0, 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


class KeptCopyListingTests(unittest.TestCase):
    def test_the_copy_is_listed_and_the_old_id_uses_it(self):
        with tempfile.TemporaryDirectory() as d:
            drive, models = Path(d) / "drive", Path(d) / "models"
            drive.mkdir()
            src = drive / "photon_v1.safetensors"
            fake_sd15(src)
            entry = model_hub.inspect_local(src)
            with mock.patch.object(paths, "MODELS_DIR", models), \
                    mock.patch.object(local_copy, "on_external_drive", lambda p: True), \
                    mock.patch.object(local_copy, "MARGIN", 0), \
                    mock.patch.dict(local_copy.KEPT, clear=True), \
                    mock.patch.object(model_hub, "LOCAL_MODELS", [entry]), \
                    mock.patch.object(model_hub, "scan_model_dirs", lambda: []):
                before = engine.all_models()
                self.assertEqual(before[entry["id"]]["file"], str(src))
                r = local_copy.copy(src)
                after = engine.all_models()
                self.assertEqual(after[entry["id"]]["file"], str(models / src.name))  # old id: loads the copy
                self.assertEqual(after[entry["id"]]["kept_as"], r["id"])
                self.assertEqual(after[r["id"]]["kept_from"], str(src))
                (models / src.name).unlink()  # the copy went to the Trash: back to the drive's file
                self.assertEqual(engine.all_models()[entry["id"]]["file"], str(src))


if __name__ == "__main__":
    unittest.main()
