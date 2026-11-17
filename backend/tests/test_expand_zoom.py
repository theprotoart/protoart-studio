"""🧠 MiniMax-H3-Image Expand: finding your picture in its zoomed-out result and lining it up."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scene(W, H, seed=3):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (H // 24, W // 24, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((W, H), Image.BICUBIC)


def _engine():
    import engine
    return engine.Engine


def test_locate_in_scaled_and_moved():
    E = _engine()
    world = _scene(1400, 1800)
    original = world.crop((400, 600, 1000, 1200))  # the 600x600 picture being expanded
    box = (100, 300, 700, 900)  # where it belongs on the 800x1200 canvas
    for k, dx, dy in ((0.85, 30, -20), (1.0, 0, 0), (1.15, -10, 25)):
        # the model drew your picture k times its size, a little off from where it belongs
        cx, cy = 400 - (box[0] + dx) / k, 600 - (box[1] + dy) / k
        drawn = world.crop((round(cx), round(cy), round(cx + 800 / k), round(cy + 1200 / k))).resize((800, 1200),
                                                                                                    Image.LANCZOS)
        placed, info = E.locate_in(drawn, original, box)
        assert info["score"] > 0.8, (k, info)
        if k <= 1.0:  # drawn smaller (or the same): scaling it up covers the canvas
            assert placed is not None, (k, info)
            got = np.asarray(placed.crop(box), np.float32)
            assert np.abs(got - np.asarray(original, np.float32)).mean() < 25, (k, info)


def test_not_found():
    placed, info = _engine().locate_in(_scene(800, 1200, 1), _scene(600, 600, 2), (100, 300, 700, 900))
    assert placed is None and info["score"] < 0.45


def test_zoom_prompt():
    E = _engine()
    p = E.zoom_out_prompt({"top": 200, "bottom": 200}, 600, 600)
    assert "taller" in p and "above and below" in p and "<Picture 1>" in p
    assert "wider" in E.zoom_out_prompt({"left": 100}, 600, 600)
