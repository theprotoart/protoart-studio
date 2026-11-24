"""🧊 Make 3D with Idol (16 GB+ Macs): the quality choices fit this Mac's memory, a splat .ply as Idol writes it reads
back as points for the viewer, the preview draws, the tensordict stand-in does what Idol's sampler needs, and the app is
wired up (no ungated FLUX.2-dev download, the Library button, the 16 GB gate)."""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import idol3d  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
         'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def _ply(path, n=4000):
    rng = np.random.default_rng(3)
    d = np.zeros((n, 17), np.float32)
    d[:, :3] = rng.uniform(-0.4, 0.4, (n, 3))
    d[:, 6:9] = rng.normal(0, 1, (n, 3))
    d[:, 9] = rng.normal(1, 1, n)
    d[:, 10:13] = np.log(0.01)
    d[:, 13] = 1
    head = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n" + "".join(
        f"property float {k}\n" for k in NAMES) + "end_header\n"
    Path(path).write_bytes(head.encode() + d.astype("<f4").tobytes())
    return d


def test_quality_fits_the_mac():
    assert idol3d.settings_for("best", 16) == (30, 65536)    # a 16 GB Mac keeps the small number of Gaussians
    assert idol3d.settings_for("best", 32) == (50, 262144)   # Idol's full default on 24 GB and more
    assert idol3d.settings_for("nonsense", 16) == idol3d.settings_for("standard", 16)
    for q in idol3d.QUALITY.values():
        for steps, n in q:
            assert n % 32 == 0 and 32768 <= n <= 1048576 and 1 <= steps <= 50


def test_ply_reads_back(tmp_path):
    import base64
    p = tmp_path / "a.ply"
    _ply(p)
    r = idol3d.read_ply(p, max_points=1000)
    assert r["total"] == 4000 and 0 < r["n"] <= 1000
    assert len(base64.b64decode(r["pos"])) == r["n"] * 12 and len(base64.b64decode(r["col"])) == r["n"] * 4


def test_preview_draws():
    d = np.random.default_rng(1).uniform(-0.4, 0.4, (3000, 3))
    img = idol3d.render_preview(d, np.full((3000, 3), 0.8), np.full(3000, 0.9), np.full(3000, 0.01), size=128)
    a = np.asarray(img)
    assert img.size == (128, 128) and (a > 150).any()  # the object shows on the dark background


def test_tensordict_stand_in():
    fake = types.ModuleType("torch")
    fake.Size = type("Size", (tuple,), {})
    fake.cat = lambda xs, dim=0: np.concatenate(xs, dim)
    ns = {}
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        exec(idol3d.SHIM, ns)
    finally:
        if saved is None:
            sys.modules.pop("torch")
        else:
            sys.modules["torch"] = saved
    TD = ns["TensorDict"]
    x = TD({"latent": np.ones((1, 4, 2)), "camera": np.zeros((1, 1, 5))}, batch_size=1)
    assert x.shape[0] == 1
    v = (1 + 3.0) * x - 3.0 * x.apply(np.zeros_like)        # Idol's guided velocity
    s = x - v * (np.float64(0.9) - np.float64(0.8))          # one Euler step (numpy times on the right)
    assert isinstance(np.float64(2.0) * x, TD)
    assert np.allclose(s["latent"], 0.6) and s["camera"].shape == (1, 1, 5)


def test_wired_up():
    src = (ROOT / "backend" / "idol3d.py").read_text()
    assert 'FLUX2_VAE = "black-forest-labs/FLUX.2-klein-4B"' in src  # the ungated klein VAE, not FLUX.2-dev's
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    assert 'engine: "idol"' in app and "splat_points" in app and "saveSplatAs" in (ROOT / "app" / "preload.js").read_text()
    assert 'id="rack-splat"' in html and "spl.status.ram_ok" in html and "splOpen(detail)" in html
    eng = (ROOT / "backend" / "engine.py").read_text()
    assert "def make_splat" in eng and 'kind="splat3d"' in eng
    assert "https://" not in html.split('id="rack-splat"')[1].split('id="rack-paper"')[0]  # no outside links
