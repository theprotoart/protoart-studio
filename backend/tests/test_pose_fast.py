"""🦴 Follow a pose on the fast engine: FLUX.2 Klein draws in the pose (image_paths), ERNIE redraws Klein's picture."""
import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("PROTOART_HOME", tempfile.mkdtemp())

import engine  # noqa: E402
from PIL import Image  # noqa: E402


class FakeMlx:
    def __init__(self):
        self.reqs = []

    def request(self, req, on_event=None, skip_event=None):
        self.reqs.append(dict(req))
        Image.new("RGB", (req["width"], req["height"]), (len(self.reqs) * 20, 0, 0)).save(req["out"])
        return {"type": "done"}


def make():
    e = object.__new__(engine.Engine)
    e.emit = lambda m: None
    e.cancel_event, e.skip_event = threading.Event(), threading.Event()
    e.previews = False
    e.mlx = FakeMlx()
    e.loads = []
    e._mlx_load = lambda spec, loras, allow_download=True, variant=None, extra_loras=None: e.loads.append((spec["family"], variant)) or 8
    e.pipe = None
    return e


class PoseFast(unittest.TestCase):
    def setUp(self):
        self.pose = Path(tempfile.mkdtemp()) / "pose.png"
        Image.new("RGB", (600, 900), "white").save(self.pose)

    def test_klein_draws_in_the_pose(self):
        e = make()
        spec = engine.all_models()["flux2_klein_mlx_q4"]
        args = {"pose_image": str(self.pose), "_raw": True}
        res = e._generate_mlx(args, "flux2_klein_mlx_q4", spec, "a knight", None, 768, 768, 4, 1.0, 2, 5, "a knight")
        self.assertEqual(len(res), 2)
        self.assertEqual(len(e.mlx.reqs), 2)  # one Klein pass per image, no second pass
        r = e.mlx.reqs[0]
        self.assertEqual(r["image_paths"][0][-4:], ".png")
        self.assertIn("a knight", r["prompt"])
        self.assertLess(r["width"], r["height"])  # the pose picture's tall shape
        self.assertEqual(res[0]["settings"]["pose"]["by"], "FLUX.2 Klein")
        self.assertEqual(e.loads, [("flux2", "edit")])

    def test_ernie_redraws_kleins_picture(self):
        e = make()
        spec = engine.all_models()["ernie_turbo_mlx_q4"] if "ernie_turbo_mlx_q4" in engine.all_models() else \
            next(v for v in engine.all_models().values() if v.get("family") == "ernie" and v.get("engine") == "mlx")
        args = {"pose_image": str(self.pose), "_raw": True, "pose_ernie_change": 0.6}
        res = e._generate_mlx(args, "ernie", spec, "a dancer", None, 1024, 1024, 8, 1.0, 1, 9, "a dancer")
        self.assertEqual(len(res), 1)
        klein, ernie = e.mlx.reqs
        self.assertIn("image_paths", klein)
        self.assertEqual(ernie["init_image"], klein["out"])
        self.assertAlmostEqual(ernie["image_strength"], 0.4)
        self.assertEqual((ernie["width"], ernie["height"]), (1024, 1024))  # ERNIE keeps its own size
        self.assertEqual([f for f, _ in e.loads], ["flux2", "ernie"])
        self.assertFalse(os.path.exists(klein["out"]))  # Klein's picture is cleaned up


if __name__ == "__main__":
    unittest.main()
