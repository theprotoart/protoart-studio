"""MOSS-Image: the model entries, and the worker's protocol with a stand-in for moss-image-mlx (no model needed)."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

FAKE = textwrap.dedent('''
    import numpy as np
    from dataclasses import dataclass, field
    @dataclass
    class Request:
        prompt: str
        negative_prompt: str = ""
        width: int = 1024
        height: int = 1024
        steps: int = 50
        guidance: float = 3.0
        seed: int = 42
        turbo: bool = False
    class MOSSImagePipeline:
        def __init__(self, model_dir, low_memory=True, cache_dir=None, loras=()):
            self.loras = list(loras); self.timings = {}
        def __call__(self, req, on_step=None):
            n = 10 if req.turbo else req.steps
            for i in range(n):
                on_step and on_step(i, n)
            return np.full((req.height, req.width, 3), 128, np.uint8)
''')


def test_models_listed_with_sizes():
    import engine
    specs = engine.MODELS
    assert specs["moss_turbo_mlx"]["family"] == "moss" and specs["moss_turbo_mlx"]["moss_turbo"]
    assert specs["moss_mlx"]["defaults"]["steps"] == 50
    import moss_engine
    assert all(w % 16 == 0 and h % 16 == 0 for w, h in moss_engine.SIZES)
    assert moss_engine.COMMIT in moss_engine.SPEC and moss_engine.SPEC.endswith(".tar.gz")


def test_worker_protocol():
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "moss_image_mlx"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(FAKE)
        mlx = Path(d) / "mlx"
        mlx.mkdir()
        (mlx / "__init__.py").write_text("")
        (mlx / "core.py").write_text("bfloat16 = 'bf16'\nfloat32 = 'f32'\n")
        out = Path(d) / "x.png"
        env = dict(os.environ, PYTHONPATH=d)
        msgs = [{"cmd": "load", "dir": d, "turbo": True, "adapters": ["s.safetensors", "d.safetensors"]},
                {"cmd": "generate", "prompt": "a sign that says HELLO", "width": 64, "height": 48, "steps": 15, "cfg": 3,
                 "seed": 5, "out": str(out)},
                {"cmd": "quit"}]
        p = subprocess.run([sys.executable, str(HERE / "moss_worker.py")], input="\n".join(json.dumps(m) for m in msgs) + "\n",
                           capture_output=True, text=True, env=env, timeout=60)
        events = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
        types = [e["type"] for e in events]
        assert types[0] == "ready" and "loaded" in types and "done" in types, (types, p.stderr[-500:])
        assert [e for e in events if e["type"] == "progress"][-1]["steps"] == 10
        from PIL import Image
        assert Image.open(out).size == (64, 48)
