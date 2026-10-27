"""✂ Cut out and 📄 Paper cut-out: the refining controls and the drawing (no model needed: a made-up matte)."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import postprocess.cutout as cut  # noqa: E402
import postprocess.papercut as paper  # noqa: E402


def _matte(h=120, w=160):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - w / 2) / 40, (yy - h / 2) / 30)
    return np.clip(1.5 - d, 0, 1).astype(np.float32)  # 1 in the middle, fading out


def test_threshold_keeps_more_when_lower():
    m = _matte()
    lo = (cut.refine(m, threshold=0.2, softness=0) > 0.5).sum()
    hi = (cut.refine(m, threshold=0.8, softness=0) > 0.5).sum()
    assert lo > hi > 0


def test_softness_makes_a_gradual_edge():
    m = _matte()
    hard = cut.refine(m, 0.5, 0)
    soft = cut.refine(m, 0.5, 0.4)
    assert set(np.unique(hard)) <= {0.0, 1.0}
    assert ((soft > 0.05) & (soft < 0.95)).sum() > ((hard > 0.05) & (hard < 0.95)).sum()


def test_grow_and_shrink():
    m = _matte()
    base = (cut.refine(m, 0.5, 0) > 0.5).sum()
    assert (cut.refine(m, 0.5, 0, grow=4) > 0.5).sum() > base > (cut.refine(m, 0.5, 0, grow=-4) > 0.5).sum()


def test_clean_bits_drops_specks_only():
    a = np.zeros((100, 100), np.float32)
    a[20:80, 20:80] = 1
    a[2:4, 2:4] = 1
    out = cut.clean_bits(a)
    assert out[3, 3] == 0 and out[50, 50] == 1


def test_compose_transparent_and_colour():
    img = Image.new("RGB", (160, 120), (200, 30, 30))
    a = cut.refine(_matte(), 0.5, 0.1)
    t = cut.compose(img, a, None)
    assert t.mode == "RGBA" and t.getpixel((0, 0))[3] == 0 and t.getpixel((80, 60))[3] == 255
    c = cut.compose(img, a, "#00ff00")
    assert c.mode == "RGB" and c.getpixel((0, 0)) == (0, 255, 0)


def test_selection_png_is_painting_sized():
    import io
    png = cut.selection_png(_matte(), 80, 60)
    im = Image.open(io.BytesIO(png))
    assert im.size == (80, 60) and im.getpixel((40, 30))[3] == 255 and im.getpixel((0, 0))[3] == 0


def test_paper_cutout_border_shadow_and_backgrounds():
    img = Image.new("RGB", (160, 120), (40, 90, 200))
    a = cut.refine(_matte(), 0.5, 0.05)
    t = paper.make(img, a, edge=6, rough=0.6, shadow=0.7)
    assert t.mode == "RGBA" and t.size == img.size
    arr = np.asarray(t)
    inside = (a > 0.5).sum()
    sticker = (arr[..., 3] > 250).sum()
    assert sticker > inside  # the paper border reaches past the subject
    assert arr[60, 80, 2] > 150  # the subject keeps its colour
    for bg in ("paper", "#ffffff", "picture"):
        out = paper.make(img, a, background=bg, print_as="poster")
        assert out.mode == "RGB"
    assert np.asarray(paper.make(img, a, print_as="mono", background="#ffffff"))[60, 80].std() < 1
