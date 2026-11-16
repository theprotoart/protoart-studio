"""🧠 MiniMax-H3-Image Expand: finding your picture in its zoomed-out result and lining it up."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scene(W, H, seed=3):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (H // 16, W // 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((W, H), Image.BICUBIC)


def test_locate_in_scaled_and_moved():
    import engine
    E = engine.Engine if hasattr(engine, "Engine") else None
    locate = (E or engine.BeeEngine).locate_in if E or hasattr(engine, "BeeEngine") else None
    world = _scene(1200, 1600)
    original = world.crop((300, 500, 900, 1100))  # 600x600, the picture being expanded
    box = (100, 300, 700, 900)  # where it goes on an 800x1200 canvas (300 added top and bottom, 100 each side)
    # the model's result: the world seen a bit closer (your picture drawn 0.9x of the size it should be) and moved
    k = 0.9
    view = world.crop((300 - 100 / k, 500 - 300 / k, 300 + 700 / k, 500 + 900 / k)) if False else None
    cx, cy = 300 - (box[0] / k), 500 - (box[1] / k)
    drawn = world.crop((round(cx), round(cy), round(cx + 800 / k), round(cy + 1200 / k))).resize((800, 1200))
    # (here the drawn picture is 1/k bigger than needed: your picture appears at k ~ 0.9 of its size... check both)
    placed, info = locate(drawn, original, box)
    assert info["score"] > 0.8, info
    if placed is not None:
        got = np.asarray(placed.crop(box), np.float32)
        assert np.abs(got - np.asarray(original, np.float32)).mean() < 20, info


def test_not_found():
    import engine
    E = engine.Engine
    placed, info = E.locate_in(_scene(800, 1200, 1), _scene(600, 600, 2), (100, 300, 700, 900))
    assert placed is None and info["score"] < 0.45


def test_zoom_prompt():
    import engine
    p = engine.Engine.zoom_out_prompt({"top": 200, "bottom": 200}, 600, 600)
    assert "taller" in p and "above and below" in p and "<Picture 1>" in p
    assert "wider" in engine.Engine.zoom_out_prompt({"left": 100}, 600, 600)
