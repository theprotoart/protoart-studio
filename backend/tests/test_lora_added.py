import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loras


def fake_lora(path):
    header = json.dumps({"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight":
                         {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


class AddedTimeTests(unittest.TestCase):
    def test_every_lora_has_when_it_was_added(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d) / "old.safetensors", Path(d) / "new.safetensors"
            fake_lora(old)
            fake_lora(new)
            long_ago = time.time() - 400 * 86400
            os.utime(new, (long_ago, long_ago))  # a copy that kept its old "modified" date still counts as new here
            rows = {r["name"]: r for r in loras.scan(d)}
            self.assertGreater(rows["new"]["added"], long_ago + 86400)
            self.assertTrue(all(isinstance(r["added"], int) and r["added"] > 0 for r in rows.values()))


if __name__ == "__main__":
    unittest.main()
