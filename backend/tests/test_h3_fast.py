"""⚡ MiniMax-H3-Image speed: Fast = 2 steps, Balanced = 3, else the usual 6; the worker switches h3-image-mlx's
few-step grid on only with fewer steps, and the few-step LoRA is the one h3-image-mlx tested."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def test_steps():
    src = (ROOT / "engine.py").read_text()
    assert 'return {"fast": 2, "balanced": 4}.get(str((args or {}).get("h3_speed") or ""), 6)' in src
    assert "not self._is_fast_lora(l[\"name\"])" in src  # never recorded among your LoRAs (Use these settings re-added it)
    assert "self._h3_steps(args, width, height)" in src and "self._h3_steps(args, draw_w, draw_h)" in src and "loras = list(loras) + [(self._h3_fast_lora(), 1.0)]" in src


def test_worker_and_lora():
    w = (ROOT / "h3_worker.py").read_text()
    assert 'if steps < 6:' in w and 'os.environ["H3_STEPS"], os.environ["H3_SHIFT"] = str(steps), "12"' in w
    e = (ROOT / "h3_engine.py").read_text()
    assert 'FAST_FILE = "minimax_h3_pdmd_2nfe_comfyui.safetensors"' in e and '"steps": int(steps)' in e
    dit = (ROOT / "vendor" / "h3_image_mlx" / "dit.py").read_text()
    assert 'steps = int(flag("H3_STEPS"))' in dit  # the vendored engine's few-step switch this relies on


def test_few_step_lora_never_doubles():
    """The pDMD LoRA in your own LoRA list is left out: Draft adds it once, Best never has it."""
    import types
    import engine
    e = engine.Engine.__new__(engine.Engine)
    sent, seen = [], {}
    e.emit = sent.append
    e._h3_python = lambda: "python"
    e._h3_fast_lora = lambda: "/hf/minimax_h3_pdmd_2nfe_comfyui.safetensors"
    e.cancel_event = types.SimpleNamespace(is_set=lambda: False)
    import h3_engine
    real = h3_engine.picture

    def fake(py, prompt, out, width, height, seed, refs, on_event=None, cancel_event=None, ref_sides=None, loras=None, steps=6):
        seen["loras"], seen["steps"] = list(loras or []), steps
        return {}
    h3_engine.picture = fake
    try:
        mine = [("/loras/minimax_h3_pdmd_2nfe_comfyui.safetensors", 1.0), ("/loras/style.safetensors", 0.8)]
        e._h3_picture("p", "/tmp/o.png", 1024, 1024, 1, loras=mine, steps=2)
        assert [l for l in seen["loras"] if "pdmd" in l[0]] == [("/hf/minimax_h3_pdmd_2nfe_comfyui.safetensors", 1.0)]
        assert ("/loras/style.safetensors", 0.8) in seen["loras"]
        e._h3_picture("p", "/tmp/o.png", 1024, 1024, 1, loras=mine, steps=6)
        assert seen["loras"] == [("/loras/style.safetensors", 0.8)] and seen["steps"] == 6
    finally:
        h3_engine.picture = real


def test_small_pictures_use_best():
    """The few-step LoRA speckles below ~2048 px: there Draft / Balanced draw with 6 steps."""
    import engine
    st = engine.Engine._h3_steps
    assert st({"h3_speed": "fast"}, 1024, 1365) == 6 and st({"h3_speed": "balanced"}, 1088, 1456) == 6
    assert st({"h3_speed": "fast"}, 2048, 2048) == 2 and st({"h3_speed": "balanced"}, 1792, 2368) == 4
    assert st({"h3_speed": "fast"}) == 2  # the setting itself (shown with the size it needs)
    app = (Path(__file__).resolve().parents[2] / "app" / "renderer" / "app.js").read_text()
    assert 'this.form.h3_best = true' in app and 'this.form.h3_side = "2048"' in app  # Draft sets its size
