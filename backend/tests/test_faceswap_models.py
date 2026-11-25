"""🔁 Face swap models: HyperSwap 1c is the default and is fed as FaceFusion feeds it (ArcFace identity as it is,
the face at 256 px in -1..1, its answer turned back from -1..1); inswapper keeps its 128 px, 0..1 path. A stand-in model
that hands the face straight back checks the colour order and scaling both ways."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faceswap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Echo:
    """An ONNX session stand-in: returns its 'target' input unchanged."""
    def __init__(self, seen):
        self.seen = seen

    def run(self, _, feeds):
        self.seen.update(feeds)
        return [feeds["target"].astype(np.float32)]


def _swapper(model):
    s = faceswap.FaceSwapper.__new__(faceswap.FaceSwapper)
    spec = faceswap.MODELS[model]
    s.model, s.size, s.licence, s.label = model, spec["size"], spec["licence"], spec["label"]
    s.in_names, s.in_types = ["target", "source"], {"target": "tensor(float)", "source": "tensor(float)"}
    s.seen = {}
    s.swapper = Echo(s.seen)
    return s


def test_hyperswap_is_default_and_licensed():
    assert faceswap.DEFAULT == "hyperswap" and faceswap.model_of(None) == "hyperswap" and faceswap.model_of("inswapper") == "inswapper"
    h = faceswap.MODELS["hyperswap"]
    assert (h["repo"], h["file"], h["size"]) == ("facefusion/models-3.3.0", "hyperswap_1c_256.onnx", 256)
    assert "ResearchRAIL" in h["licence"]


def test_round_trip_both_models():
    rng = np.random.default_rng(0)
    for model, n in (("hyperswap", 256), ("inswapper", 128)):
        s = _swapper(model)
        crop = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)  # BGR
        out = s._run(crop, np.ones((1, 512), np.float32))
        assert out.shape == (n, n, 3) and np.abs(out.astype(int) - crop.astype(int)).max() <= 1, model
        t = s.seen["target"]
        if model == "hyperswap":
            assert t.min() >= -1.0001 and t.max() <= 1.0001 and t.min() < -0.5  # -1..1
        else:
            assert t.min() >= 0 and t.max() <= 1.0001
        assert np.allclose(t[0, 0], crop[:, :, 2] / (127.5 if model == "hyperswap" else 255.0) - (1 if model == "hyperswap" else 0), atol=1e-4)  # RGB order


def test_template_matches_facefusion_arcface_128():
    ff = np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                   [0.38710391, 0.72160547], [0.61507734, 0.72034453]])
    assert np.allclose(faceswap.template(256) / 256, ff, atol=1e-5)


def test_wired_up():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    eng = (ROOT / "backend" / "engine.py").read_text()
    assert 'swapper: sw.model || "hyperswap"' in app and 'model: "hyperswap" },' in app
    assert 'fsw.model_of(args.get("swapper"))' in eng and '"licence": self.face_swapper.licence' in eng
