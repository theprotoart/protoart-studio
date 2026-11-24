"""⚡ MiniMax-H3-Image speed: Fast = 2 steps, Balanced = 3, else the usual 6; the worker switches h3-image-mlx's
few-step grid on only with fewer steps, and the few-step LoRA is the one h3-image-mlx tested."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def test_steps():
    src = (ROOT / "engine.py").read_text()
    assert 'return {"fast": 2, "balanced": 3}.get(str((args or {}).get("h3_speed") or ""), 6)' in src
    assert "steps=self._h3_steps(args)" in src and "loras = list(loras or []) + [(self._h3_fast_lora(), 1.0)]" in src


def test_worker_and_lora():
    w = (ROOT / "h3_worker.py").read_text()
    assert 'if steps < 6:' in w and 'os.environ["H3_STEPS"], os.environ["H3_SHIFT"] = str(steps), "12"' in w
    e = (ROOT / "h3_engine.py").read_text()
    assert 'FAST_FILE = "minimax_h3_pdmd_2nfe_comfyui.safetensors"' in e and '"steps": int(steps)' in e
    dit = (ROOT / "vendor" / "h3_image_mlx" / "dit.py").read_text()
    assert 'steps = int(flag("H3_STEPS"))' in dit  # the vendored engine's few-step switch this relies on
