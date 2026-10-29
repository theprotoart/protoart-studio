"""🧩 Combine and 🧅 Peel layers: light matching, placing, the prompt, peeling and the .psd file (no model needed)."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import postprocess.compose as comp  # noqa: E402


def _subject(w=60, h=100, colour=(200, 200, 200)):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    im.paste(colour + (255,), (10, 10, w - 10, h - 5))
    return im


def test_match_light_moves_towards_the_scene():
    sub = _subject()
    dark_blue = Image.new("RGB", (200, 120), (20, 30, 90))
    out = comp.match_light(sub, dark_blue, 0.8)
    before, after = sub.getpixel((30, 50)), out.getpixel((30, 50))
    assert sum(after[:3]) < sum(before[:3])          # darker
    assert after[2] > after[0]                         # bluer
    assert out.getpixel((0, 0))[3] == 0                # the see-through part stays see-through
    assert comp.match_light(sub, dark_blue, 0).getpixel((30, 50)) == before


def test_match_light_uses_the_region_around_it():
    scene = Image.new("RGB", (200, 100), (240, 240, 240))
    scene.paste((10, 10, 10), (100, 0, 200, 100))
    sub = _subject(colour=(128, 128, 128))
    left = comp.match_light(sub, scene, 1, region=(0, 0, 100, 100)).getpixel((30, 50))
    right = comp.match_light(sub, scene, 1, region=(100, 0, 200, 100)).getpixel((30, 50))
    assert sum(left[:3]) > sum(right[:3])


def test_fit_box_keeps_shape_and_stands_on_the_bottom():
    l, t, w, h = comp.fit_box((50, 100), {"x": 0.5, "y": 0, "w": 0.5, "h": 1}, (200, 100))
    assert (w, h) == (50, 100) and t + h == 100 and l == 125


def test_place_and_blend_area():
    scene = Image.new("RGB", (300, 200), (0, 120, 0))
    out, cover = comp.place(scene, [(_subject(), {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.6})])
    assert out.size == scene.size and np.asarray(cover).max() == 255
    assert out.getpixel((0, 0)) == (0, 120, 0)
    area = np.asarray(comp.blend_area(cover))
    assert area.sum() > np.asarray(cover).sum()        # room for its shadow and edges
    assert area[0, 299] == 0                           # far away stays untouched


def test_prompt_numbers_the_pictures():
    p = comp.combine_prompt("sitting on the bench", 1)
    assert "picture 2" in p and "Sitting on the bench." in p and "lighting" in p
    p2 = comp.combine_prompt("", 2, placed=True)
    assert "picture 3" in p2 and "already placed" in p2


def test_peel_lifts_the_painted_part():
    img = Image.new("RGB", (120, 80), (10, 200, 10))
    img.paste((220, 20, 20), (40, 20, 80, 60))
    mask = Image.new("L", img.size, 0)
    mask.paste(255, (40, 20, 80, 60))
    layer, hole = comp.peel(img, mask, feather=0)
    assert layer.getpixel((60, 40)) == (220, 20, 20, 255) and layer.getpixel((5, 5))[3] == 0
    h = np.asarray(hole)
    assert h[40, 60] == 255 and h[40, 39] == 255 and h[5, 5] == 0   # a little past the edge
    try:
        comp.peel(img, Image.new("L", img.size, 0))
        assert False
    except ValueError:
        pass


def test_peel_follows_the_matte_inside_the_selection():
    img = Image.new("RGB", (60, 60), (50, 50, 50))
    mask = Image.new("L", img.size, 0)
    mask.paste(255, (10, 10, 50, 50))
    matte = np.zeros((60, 60), np.float32)
    matte[20:40, 20:40] = 1
    matte[0:5, 0:5] = 1                                # outside the selection: ignored
    layer, _ = comp.peel(img, mask, feather=0, matte=matte)
    a = np.asarray(layer.getchannel("A"))
    assert a[30, 30] == 255 and a[12, 12] == 0 and a[2, 2] == 0


def test_psd_is_readable_with_its_layers(tmp_path):
    W, H = 64, 48
    bg = Image.new("RGBA", (W, H), (30, 90, 200, 255))
    red = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    red.paste((255, 0, 0, 255), (10, 8, 30, 40))
    p = tmp_path / "x.psd"
    comp.write_psd(p, (W, H), [("Background", bg, True), ("Rød", red, True), ("hidden", red, False)])
    im = Image.open(p)
    assert im.size == (W, H) and im.getpixel((15, 15)) == (255, 0, 0) and im.getpixel((50, 5)) == (30, 90, 200)
    names = [l[0] for l in im.layers]
    assert names[0] == "Background" and len(names) == 3
    assert im.layers[1][2] == (10, 8, 30, 40)          # each layer cropped to what's on it


def test_combine_prepare_places_and_limits_the_change(tmp_path):
    import engine
    eng = engine.Engine.__new__(engine.Engine)
    eng.emit = lambda m: None
    matte = np.zeros((100, 60), np.float32)
    matte[10:95, 10:50] = 1
    eng._cut_matte = lambda src, image, emit: matte
    scene = Image.new("RGB", (320, 200), (90, 120, 160))
    ref = Image.new("RGB", (60, 100), (250, 250, 250))
    ref.paste((200, 60, 40), (10, 10, 50, 95))
    image, refs, mask, prompt = eng._combine_prepare(
        scene, ["/x/ref.png"], [ref], {"cutout": True, "match_light": True, "place": [{"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.7}]},
        "waving")
    assert image.size == scene.size and image.getpixel((0, 0)) == (90, 120, 160)
    assert image.getpixel((160, 120)) != (90, 120, 160)          # the character is in
    assert mask is not None and mask.getpixel((5, 5)) == 0 and mask.getpixel((160, 120)) > 200
    assert refs[0].getpixel((0, 0)) == (255, 255, 255)            # its old background is gone from the reference
    assert "already placed" in prompt and "Waving" in prompt
    # left to decide: the scene goes in as it is, nothing is limited
    image2, _, mask2, prompt2 = eng._combine_prepare(scene, ["/x/ref.png"], [ref], {"cutout": False}, "")
    assert mask2 is None and image2 is scene and "Add the character from picture 2" in prompt2
    try:
        eng._combine_prepare(scene, [], [], {}, "")
        assert False
    except engine.UserError:
        pass
