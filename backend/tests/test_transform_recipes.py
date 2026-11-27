"""🎭 Transform a person: every preset has its boxes, and every [PLACEHOLDER] in its prompt has a box that fills it."""
import re
from pathlib import Path

APP = (Path(__file__).resolve().parents[2] / "app" / "renderer" / "app.js").read_text()
BLOCK = APP[APP.index("const EDIT_RECIPES = ["):APP.index("\n];", APP.index("const EDIT_RECIPES = ["))]


def _presets():
    return re.findall(r'\{ id: "(tf_\w+)".*?slots: \[(.*?)\],\n    prompt: "(.*?)",\n', BLOCK, re.S)


def _nostyle():
    return set(re.findall(r'\{ id: "(tf_\w+)"[^\n]*\n[^\n]*nostyle: true', BLOCK))


def test_presets():
    ids = [p[0] for p in _presets()]
    assert len(ids) == 64 and len(set(ids)) == 64  # 🎨 style transfer + 27 looks + 23 ✨ photo effects + 🍬 sweet treat + 10 🛠 restore & technical + 🔬 Ultra-detail + 🔤 Fill a shape
    for pid, slots, prompt in _presets():
        assert 'kind: "style"' in slots or pid in _nostyle(), pid
        for ph in re.findall(r"\[[A-Z0-9 ]+\]", prompt):
            assert f'ph: "{ph}"' in slots, f"{pid}: {ph} has no box"
        for m in re.finditer(r'kind: "item"[^}]*', slots):
            assert "ph:" in m.group(0) or "add:" in m.group(0), f"{pid}: an item box that goes nowhere"
    for tm in ("lego", "gta", "grand theft", "rockstar", "playstation", "sony", "pixar", "disney", "marvel", "street fighter", "pokemon", "funko", "naruto", "demon slayer", "tanjiro", "ghibli"):  # no trademark names
        assert tm not in BLOCK.lower(), tm
