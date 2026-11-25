"""🎞 Film looks (Tools): every look runs and keeps the picture's size (and see-through parts), strength 0 changes
nothing, the black-and-white film is grey, the night film glows red round lights, grain is grain (repeatable, and gone
at 0), the looks differ, no brand names, and the module is wired into Tools and the patchbay."""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import postprocess.film as film  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _pic(w=320, h=240):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([xx / w * 200 + 30, yy / h * 180 + 40, 120 + 60 * np.sin(xx / 20)], -1)
    a = a + rng.normal(0, 3, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_every_look_runs():
    im = _pic()
    outs = {}
    for k in film.ORDER:
        out = film.apply(im, k)
        assert out.size == im.size and out.mode == "RGB"
        outs[k] = np.asarray(out, dtype=np.float32)
    keys = list(outs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert np.abs(outs[a] - outs[b]).mean() > 3, (a, b)  # each look is its own


def test_strength_zero_is_the_picture():
    im = _pic()
    assert np.array_equal(np.asarray(film.apply(im, "slide", strength=0)), np.asarray(im))


def test_black_and_white():
    a = np.asarray(film.apply(_pic(), "gritty_bw"), dtype=int)
    assert np.abs(a[..., 0] - a[..., 1]).max() <= 1 and np.abs(a[..., 1] - a[..., 2]).max() <= 1


def test_night_film_glows_red_round_lights():
    im = Image.new("RGB", (300, 300), (20, 24, 30))
    im.paste((255, 255, 255), (140, 140, 160, 160))  # a lamp in the dark
    with_glow = np.asarray(film.apply(im, "night_tungsten", grain=0), dtype=float)
    without = np.asarray(film.apply(im, "night_tungsten", grain=0, halation=0), dtype=float)
    ring = (slice(120, 135), slice(140, 160))  # just outside the lamp
    extra = with_glow[ring].mean((0, 1)) - without[ring].mean((0, 1))
    assert extra[0] > 8 and extra[0] > 2 * extra[2]  # redder round the light
    far = (slice(0, 30), slice(0, 30))
    assert abs(with_glow[far].mean() - without[far].mean()) < 1.5  # not everywhere


def test_grain():
    im = Image.new("RGB", (200, 200), (128, 128, 128))
    a = np.asarray(film.apply(im, "gritty_bw", seed=3), dtype=float)
    b = np.asarray(film.apply(im, "gritty_bw", seed=3), dtype=float)
    none = np.asarray(film.apply(im, "gritty_bw", grain=0), dtype=float)
    assert np.array_equal(a, b)                         # the same picture twice is the same
    assert a[40:160, 40:160].std() > 3 > none[40:160, 40:160].std() + 0.5


def test_see_through_kept():
    im = _pic().convert("RGBA")
    im.putalpha(Image.new("L", im.size, 90))
    out = film.apply(im, "disposable")
    assert out.mode == "RGBA" and np.asarray(out)[..., 3].max() == 90


def test_no_brand_names_and_wired_up():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    looks = app[app.index("const FILM_LAB = ["):app.index("];", app.index("const FILM_LAB = ["))]
    code = (ROOT / "backend" / "postprocess" / "film.py").read_text()
    for name in ("kodak", "fuji", "portra", "superia", "ektachrome", "tri-x", "trix", "cinestill", "ilford"):
        for text in (looks.lower(), code.lower()):
            assert not re.search(r"\b" + re.escape(name) + r"\b", text), name
    assert re.findall(r'id: "(\w+)"', looks) == film.ORDER
    assert '{ id: "film", label: "FILM"' in app and 'id === "film"' in app and 'film: ["film"]' in app
    assert 'id="rack-film"' in html and "toolFilm" in html and "patchPicked('film')" in html
    bee = (ROOT / "backend" / "bee_backend.py").read_text()
    assert 'cmd == "film_preview"' in bee and 'kind == "film"' in bee and '"film", "peel", "peel_save")' in bee
