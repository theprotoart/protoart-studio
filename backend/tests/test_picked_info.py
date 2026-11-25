"""The picture info names what was picked: a recipe's own choices (Fill a shape, Seamless pattern, a look's boxes),
MiniMax-H3's speed and layout preset, Tools' film look."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_recipe_opts_kept_short():
    ro = engine.recipe_opts({"recipe_opts": {"Shape": "the word “LOVE”", "Inverted": "", "x" * 50: "y" * 200}})
    assert ro["Shape"] == "the word “LOVE”" and "Inverted" not in ro
    assert all(len(k) <= 30 and len(v) <= 80 for k, v in ro.items())
    assert engine.recipe_opts({}) is None and engine.recipe_opts({"recipe_opts": "nope"}) is None


def test_h3_layout_preset_named():
    assert engine.look_info({"h3_preset": "📓 Manga draft page"}) == {"h3_preset": "📓 Manga draft page"}
    assert engine.look_info({"h3_preset": "any"}) == {}


def test_wired_up():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    eng = (ROOT / "backend" / "engine.py").read_text()
    assert "pickedInfo(detail)" in html and "    pickedInfo(d) {" in app
    assert app.count("recipe_opts: this.shapeOpts(") == 2 and 'recipe: "tf_pattern"' in app and "this.tfOpts(rec)" in app
    assert 'settings["recipe_opts"] = recipe_opts(args)' in eng and '"h3_speed": {2: "draft", 4: "balanced"}' in eng


def test_use_these_settings_brings_edits_back():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    eng = (ROOT / "backend" / "engine.py").read_text()
    assert 'if (["word_edit", "combined", "camera_angle"].includes(r.kind)) return this.reuseEdit(r);' in app
    assert app.count("recipe_state: ") >= 3 and "    shapeState() {" in app
    for key in ('settings["init_image"]', 'settings["user_prompt"]', 'settings["edit_refs"]', 'settings["recipe_state"]', 'settings["h3_side"]'):
        assert key in eng, key
    assert 'cmd == "file_exists"' in (ROOT / "backend" / "bee_backend.py").read_text()


def test_no_2048_edits_under_16gb_and_eta_doesnt_freeze():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    assert "h3_speed: this.h3SpeedEdit() ?" in app and "this.sys.ram_gb >= 15.5" in app
    assert "elapsed * pr.steps / Math.max(1, pr.step)" in app and "taking longer than usual: almost done…" in app
