"""✨ Effects lab and the 1.6.31 extras: the sparkler follows the painted path's centre line, the engraving is ink on
paper that darkens with the picture, the fluted glass changes only its panel; Fill a shape's threshold / solid silhouette;
the rembg models fall back to a Hugging Face copy that must match rembg's own checksums; everything is wired up."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import postprocess.fx as fx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _pic(w=320, h=240):
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([xx / w * 220 + 20, yy / h * 200 + 30, 140 + 0 * xx], -1)
    return Image.fromarray(a.astype(np.uint8))


def test_thin_gives_a_centre_line():
    m = np.zeros((60, 120), bool)
    m[20:41, 10:110] = True                      # a stroke 21 px thick
    t = fx.thin(m)
    assert t[:, 30:90].sum(0).max() <= 2 and t[:, 30:90].any()   # about one pixel across
    assert abs(np.nonzero(t[:, 60])[0].mean() - 30) <= 1          # in the middle


def test_sparkler_lights_the_path():
    im = _pic()
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).line([(40, 120), (280, 120)], fill=255, width=12)
    out = np.asarray(fx.sparkler(im, mask, night=True), float)
    src = np.asarray(im, float)
    assert out[120, 60:260].mean() > 235                              # white-hot along the line
    assert out[10:40, 10:60].mean() < src[10:40, 10:60].mean() * 0.6  # night: the rest darker
    try:
        fx.sparkler(im, Image.new("L", im.size, 0))
        raise AssertionError("an empty path must be refused")
    except ValueError:
        pass


def test_engrave_is_ink_on_paper():
    im = _pic()
    for style in ("lines", "dots"):
        a = np.asarray(fx.engrave(im, style=style), float)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        assert np.abs(r - g).max() < 30 and a.min() < 60 and a.max() > 230   # ink and paper only
        dark, light = a[:, :60].mean(), a[:, -60:].mean()
        assert dark < light                                                  # more ink where the picture is darker


def test_glass_only_inside_its_panel():
    im = _pic(400, 300)
    out = np.asarray(fx.glass(im, "circle", 0.5, 0.5, 0.4), float)
    src = np.asarray(im, float)
    assert np.abs(out[:40, :40] - src[:40, :40]).max() < 2      # outside: untouched
    assert np.abs(out[130:170, 180:220] - src[130:170, 180:220]).mean() > 4  # inside: changed


def test_shape_threshold_and_solid():
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(prefix="protoart_test_"))
    import bee_backend as b
    im = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 160, 160), fill=(60, 60, 60))      # a dark disc…
    d.ellipse((80, 80, 120, 120), fill=(200, 200, 200))   # …with a light middle
    src = tmp_path / "s.png"
    im.save(src)
    def black(r):
        return (np.asarray(Image.open(r["path"]).convert("L")) < 128).mean()
    hole = black(b.handle_quick("make_shape", {"path": str(src), "dark": False, "threshold": 128}))
    more = black(b.handle_quick("make_shape", {"path": str(src), "dark": False, "threshold": 230}))
    solid = b.handle_quick("make_shape", {"path": str(src), "dark": False, "threshold": 128, "solid": True})
    assert more > hole and black(solid) > hole and "solid" in solid["how"]
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_rembg_falls_back_to_a_checked_copy():
    import shutil
    import tempfile
    import types
    import urllib.request
    import postprocess.cutout as cut
    tmp_path = Path(tempfile.mkdtemp(prefix="protoart_test_"))
    saved = (cut._rembg_path, sys.modules.get("net"), urllib.request.urlopen, sys.modules.get("huggingface_hub"),
             cut.REMBG_SHA256["u2net_human_seg"])
    try:
        cut._rembg_path = lambda name: tmp_path / "rembg" / f"{name}.onnx"
        fake_net = types.ModuleType("net")
        fake_net.CURL = None
        sys.modules["net"] = fake_net
        def offline(*a, **k):
            raise OSError("Could not resolve host: github.com")
        urllib.request.urlopen = offline
        good = tmp_path / "good.onnx"
        good.write_bytes(b"x" * 2_000_000)
        cut.REMBG_SHA256["u2net_human_seg"] = cut._sha256(good)
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda repo, f: (str(good) if (repo, f) == (cut.REMBG_HF, "u2net_human_seg.onnx") else 1 / 0)
        sys.modules["huggingface_hub"] = hub
        assert cut._fetch_rembg("u2net_human_seg").read_bytes() == good.read_bytes()
        cut.REMBG_SHA256["u2net_human_seg"] = "0" * 64          # a copy that isn't the real file
        (tmp_path / "rembg" / "u2net_human_seg.onnx").unlink()
        try:
            cut._fetch_rembg("u2net_human_seg")
            raise AssertionError("a file with the wrong checksum must be refused")
        except RuntimeError as e:
            assert "checksum" in str(e)
        assert not (tmp_path / "rembg" / "u2net_human_seg.onnx").exists()
    finally:
        cut._rembg_path, net0, urllib.request.urlopen, hub0, cut.REMBG_SHA256["u2net_human_seg"] = saved
        for k, v in (("net", net0), ("huggingface_hub", hub0)):
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_wired_up():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    bee = (ROOT / "backend" / "bee_backend.py").read_text()
    eng = (ROOT / "backend" / "engine.py").read_text()
    assert '{ id: "fx", label: "FX"' in app and 'id === "fx"' in app and 'fx: ["fx"]' in app and "sparkle:" in app
    assert 'id="rack-fx"' in html and "toolPaintStart(fxPaintKind())" in html and "['paint', '🖌 Painted area']" in html
    assert 'cmd == "fx_preview"' in bee and 'kind == "fx"' in bee and "def fx(self, args" in eng
    assert "const TEXT_FX = [" in app and "TEXT_FX_PROMPT" in app and "pickTextFx(t.id)" in html
    assert "shapeRemake()" in html and '"threshold"' in bee and 'a.get("solid")' in bee
    assert "tile_thumb" not in bee and "setTileThumb" not in html  # tile pictures are the app's own; nobody can change them
    styles = (ROOT / "app" / "renderer" / "zimage_styles.js").read_text()
    assert styles.count('"cat": "Effects"') == 24 and "'Effects']" in html
    for gone in ('label: "🧍 Turnaround sheet"', 'label: "📖 Comic panel"', 'label: "🌴 Crime-game art"'):
        assert gone not in app, gone   # duplicates of 🎭 Transform looks
    for t in ("tf_fx_sparkler", "tf_fx_glass"):
        assert (ROOT / "app" / "renderer" / "assets" / "presets" / f"{t}.jpg").exists()


def test_batch_run_every_look():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    body = app[app.index("    batchLooks(which"):app.index("    tfSlotLabel(sl)")]
    assert 'r.id !== "tf_ultra_detail"' in body and "this.editRecipeLora(r)" in body and "(!r.klein || klein)" in body
    assert "tileMaking.stop" in body and "loras: [{ name: lora.name" in body  # LoRA looks with their LoRA
    assert 'which === "star" ? this.tfStars.includes(r.id)' in body and "tile_thumb" not in body
    assert "Run every look on one photo" in html and 'tfStar(r.id)' in html and "lookBatch.which = 'star'" in html
    assert "lookBatch.which = 'fx'" not in html and "lookBatch.tiles" not in html


def test_dispersion_direction():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    blk = app[app.index('id: "tf_fx_dispersion"'):][:3000]
    assert 'kind: "choice"' in blk and '["left", "← Left"]' in blk and 'dflt: "left"' in blk
    assert 'if (sl.kind === "choice")' in app and "sl.kind === 'choice'" in html
