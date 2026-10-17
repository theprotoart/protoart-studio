"""Object selection (GrabCut): finds a plain object inside the box."""
import base64
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SelectObjectTests(unittest.TestCase):
    def test_finds_disc(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("no OpenCV")
        import select_object
        img = Image.new("RGB", (200, 160), (40, 120, 200))
        rng = np.random.default_rng(0)
        noise = (rng.random((160, 200, 3)) * 30).astype(np.uint8)
        img = Image.fromarray(np.clip(np.asarray(img) + noise, 0, 255).astype(np.uint8))
        ImageDraw.Draw(img).ellipse((70, 50, 130, 110), fill=(230, 60, 40))
        url, found = select_object.select(img, (0.25, 0.2, 0.5, 0.6), 200, 160)
        self.assertTrue(found)
        m = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))[..., 3] > 0
        self.assertTrue(m[80, 100])          # centre of the disc picked
        self.assertFalse(m[10, 10])          # background outside the box not picked
        self.assertFalse(m[60, 55])          # background inside the box not picked
        area = m.sum()
        self.assertLess(abs(area - 3.1416 * 30 * 30) / (3.1416 * 30 * 30), 0.25)




class SideLaneTests(unittest.TestCase):
    def test_heal_runs_beside_a_job(self):
        try:
            import bee_backend as bb
        except Exception as e:
            self.skipTest(str(e))
        self.assertIn("heal", bb.SIDE_LANE)
        sent = []

        class Eng:
            def heal(self, a, emit=None):
                emit({"type": "status", "msg": "Removing what you painted…"})
                return {"id": 7, "kind": "healed"}

        old_engine, old_emit = bb.engine, bb.emit
        bb.engine, bb.emit = (lambda: Eng()), sent.append
        try:
            bb.run_side(42, "heal", {})
        finally:
            bb.engine, bb.emit = old_engine, old_emit
        self.assertEqual(sent[0]["job"], 42)  # its own progress, not the running job's
        self.assertEqual(sent[-1], {"type": "result", "id": 42, "data": {"id": 7, "kind": "healed"}})


if __name__ == "__main__":
    unittest.main()
