"""🔁 Seamless pattern: the shifted picture's edges join up when it repeats, the repainted cross stays away from the
edges (so the join can't come back), the wording covers each use, and the app is wired up."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pattern  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _not_tiling(w=640, h=480):
    x = np.linspace(0, 255, w)[None, :].repeat(h, 0)
    y = np.linspace(0, 255, h)[:, None].repeat(w, 1)
    noise = np.random.default_rng(2).normal(0, 6, (h, w))
    a = np.clip(np.stack([x, y, (x + y) / 2], 2) + noise[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def test_offset_joins_the_edges():
    im = _not_tiling()
    assert pattern.seam_score(im) > 10                      # left/right and top/bottom don't match: a hard seam
    shifted = pattern.offset_half(im)
    assert pattern.seam_score(shifted) < 3                  # the new edges were neighbours in the middle
    assert pattern.offset_half(pattern.offset_half(im)).tobytes() == im.tobytes()  # shifting twice is the original


def test_prepare_leaves_the_edges_alone(tmp_path):
    im = _not_tiling()
    src = tmp_path / "a.png"
    im.save(src)
    r = pattern.prepare(src, tmp_path, "t")
    out, m = Image.open(r["path"]), np.asarray(Image.open(r["mask"]))
    assert out.size == im.size and m.shape == (im.height, im.width)
    edge = 40
    for band in (m[:edge], m[-edge:], m[:, :edge], m[:, -edge:]):
        assert band[band > 0].size < band.size               # the cross crosses the edges only in its middle
    assert m[:, :edge][:edge].max() == 0 and m[-edge:, -edge:].max() == 0  # the corners are never repainted
    assert m[im.height // 2, im.width // 2] == 255
    ref = np.asarray(pattern.offset_half(im)).astype(int)
    got = np.asarray(out).astype(int)
    outside = m == 0
    assert np.abs(got - ref)[outside].max() <= 1            # outside the cross the picture is untouched
    assert pattern.seam_score(out) < 3


def test_wording():
    for use in pattern.USES:
        p = pattern.first_prompt("words", use, "lemons")
        assert "lemons" in p and "seamless" in p and "No border" in p
    assert "Picture 1" in pattern.first_prompt("inspired", "fabric", "")
    assert "blurred band" in pattern.SEAM_PROMPT


def test_wired_up():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    bee = (ROOT / "backend" / "bee_backend.py").read_text()
    assert 'id: "tf_pattern"' in app and "patternRun()" in app and '"pattern_prep"' in app
    assert 'cmd == "pattern_prep"' in bee and 'cmd == "pattern_prompt"' in bee
    assert "pattern-card" in html and "pattern-tiles" in html
    assert "https://" not in html.split("pattern-card")[1].split("</template>\n          </div>")[0]
