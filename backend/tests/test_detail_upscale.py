"""✨ Detail upscale (Z-Image Turbo, tile by tile): tiles cover the picture and blend back without seams."""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def test_detail_upscale_tiles_and_blends():
    import tempfile
    import engine
    import thumbs
    tmp_path = Path(tempfile.mkdtemp())
    undo = [(engine, "total_ram_gb", engine.total_ram_gb), (engine.paths, "HOME", engine.paths.HOME), (thumbs, "make", thumbs.make)]
    rng = np.random.default_rng(1)
    src = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))  # detailed everywhere: every tile redrawn
    calls = []

    class Mlx:
        def request(self, req, on_event=None):  # the "redraw": the tile as it came in
            calls.append(req)
            Image.open(req["init_image"]).save(req["out"])

    class Lib:
        def add_image(self, path, settings, parent_id=None, kind=None):
            return {"id": 1, "path": str(path), "settings": settings}

    saved = {}

    class Fake:
        DETAIL_MODEL = engine.Engine.DETAIL_MODEL
        cancel_event = threading.Event()
        mlx = Mlx()
        library = Lib()
        def _source_picture(self, a): return {"id": 7}, src
        def emit(self, ev): pass
        def wait_if_paused(self): pass
        def device(self): return "cpu"
        def _mlx_load(self, spec, reqs, extra_loras=None): saved["loras"] = extra_loras
        def _save(self, img, settings, suffix=None):
            saved["img"] = img
            return tmp_path / "out.png"
        def _content_check(self, r, img): return r

    engine.total_ram_gb = lambda: 8
    engine.paths.HOME = tmp_path
    thumbs.make = lambda *a, **k: None
    try:
        rec = engine.Engine.detail_upscale(Fake(), {"scale": 2, "seed": 3}, flow_dpo="/x/zit_fdpo_v1.safetensors")
    finally:
        for mod, name, val in undo:
            setattr(mod, name, val)
    out = saved["img"]
    assert out.size == (1280, 960)
    assert len(calls) >= 4 and all(c["width"] == 768 and abs(c["image_strength"] - 0.7) < 1e-6 for c in calls)
    assert saved["loras"] == [("/x/zit_fdpo_v1.safetensors", 0.6)]
    ref = np.asarray(src.resize(out.size, Image.LANCZOS), np.float32)
    assert np.abs(np.asarray(out, np.float32) - ref).mean() < 8  # an unchanged redraw blends back to the picture
    assert rec["settings"]["detail_upscale"]["tiles"] == len(calls)


def test_crop_around_painted_area():
    import engine
    assert engine.crop_around((500, 500, 560, 540), (2000, 1500)) == (338, 328, 722, 712)  # small: 384 px around it
    assert engine.crop_around((0, 0, 1800, 1400), (2000, 1500)) is None  # big: the whole picture is edited
    left, top, right, bottom = engine.crop_around((1950, 1450, 2000, 1500), (2000, 1500))
    assert right == 2000 and bottom == 1500 and right - left == 384  # kept inside the picture
    assert engine.crop_around(None, (100, 100)) is None


def test_plain_tiles_are_left_alone():
    import engine
    sky = Image.new("RGB", (512, 512), (120, 160, 220))
    assert engine.plain_tile(sky)
    rng = np.random.default_rng(2)
    busy = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
    assert not engine.plain_tile(busy)
