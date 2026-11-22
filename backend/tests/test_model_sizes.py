"""Every model type in the catalog comes with its own size buttons, shapes and sampler picks (so a new model can't be
added without them), the shapes are sizes the app can make, and Create's 🎚 Denoise is wired up."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import samplers  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "app" / "renderer" / "app.js").read_text()
ENGINE = (ROOT / "backend" / "engine.py").read_text()
HTML = (ROOT / "app" / "renderer" / "index.html").read_text()
FAMILIES = set(re.findall(r'"family": "(\w+)"', ENGINE))


def _block(start, end):
    i = APP.index(start)
    return APP[i:APP.index(end, i)]


def test_every_model_type_has_sizes():
    tiers = _block("    sizeTiers() {", "    currentTier() {")
    shapes = _block("    sizePresets() {", "    livePreviewShown() {")
    for f in FAMILIES:
        assert re.search(rf'\b{f}: \[|family === "{f}"', tiers), f"{f}: no Small / Medium / Large sizes in sizeTiers"
        assert re.search(rf"\b{f}: \[|sets\.{f} =", shapes), f"{f}: no shapes in sizePresets"


def test_every_model_type_has_sampler_picks():
    picks = _block("const SAMPLER_PICKS = {", "\n};")
    for f in FAMILIES - {"anima"}:  # Anima's pipeline picks its own
        assert re.search(rf"\b{f}: \{{", picks), f"{f}: no entry in SAMPLER_PICKS"
        if f not in ("sd15", "sdxl"):
            ids = {s["id"] for s in samplers.list_for(f)}
            assert "flow_euler" in ids, f"{f}: the backend offers no flow sampler"


def test_shapes_are_makeable():
    shapes = _block("    sizePresets() {", "    livePreviewShown() {")
    for w, h in re.findall(r'\["[^"]+", (\d+), (\d+)\]', shapes):
        w, h = int(w), int(h)
        assert 256 <= w <= 2048 and 256 <= h <= 2048, (w, h)  # the width / height boxes go up to 2048
        assert w % 8 == 0 and h % 8 == 0, (w, h)


def test_denoise_in_create():
    assert "dn: { image:" in APP and "DENOISE_FAMILIES" in APP
    fams = re.search(r"const DENOISE_FAMILIES = \[([^\]]*)\]", APP).group(1)
    for f in re.findall(r'"(\w+)"', fams):
        assert f in FAMILIES, f"{f} in DENOISE_FAMILIES is not a catalog model type"
    for f in ("moss", "h3", "flux2", "flux3", "fibo2"):  # these can't start from a picture in Create
        assert f'"{f}"' not in fams
    args = _block("    createArgs(overrides) {", "    cleanExtras(args) {")
    assert "init_image: dn.image" in args and "Number(dn.value) < 1" in args
    assert 'v-if="denoiseOk"' in HTML and "onDnDrop" in HTML
