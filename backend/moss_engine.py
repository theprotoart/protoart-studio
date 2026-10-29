"""MOSS-Image (OpenMOSS, Apache-2.0) on the Apple GPU: a 5B text-to-image model that is especially good at lettering
(signs, posters, book covers, infographics) and layouts.

It runs with moss-image-mlx (felix-ab, Apache-2.0), a pure-MLX port checked against the official pipeline, in its
own Python environment (~/.protoartstudio/venv-moss: it needs newer libraries than the other engines), installed
once at a pinned commit. The worker (moss_worker.py) talks JSON lines, like the fast engine's.

Weights: VisualInference/MOSS-Image-SFT-MLX-compact (8-bit transformer, 6-bit Qwen3.5-9B text encoder, FLUX.2
small VAE decoder; ~12 GB) at a pinned revision. Turbo: a trained few-step student (VisualInference/
MOSS-Image-SFT-Turbo-MLX) plus nynxz's destripe LoRA, 15 passes instead of 100.
"""
import os
import shutil
import sys
from pathlib import Path

import mlx_engine
from mlx_engine import EngineError, CancelledError  # noqa: F401  (re-exported for engine.py)

HERE = Path(__file__).resolve().parent
_HOME = Path(os.environ.get("PROTOART_HOME") or (Path.home() / ".protoartstudio"))
VENV = _HOME / "venv-moss"
PY = VENV / "bin" / "python"
MARKER = VENV / ".protoart-installed"
COMMIT = "da75091341702bdd14c07bb2ce72705abc25724e"   # moss-image-mlx 0.2.0, pinned: the code can't change under us
# the source archive of that exact commit (no git needed on the Mac)
SPEC = f"moss-image-mlx @ https://github.com/felix-ab/moss-image-mlx/archive/{COMMIT}.tar.gz"
INSTALL_GB = 1.0

REPO = "VisualInference/MOSS-Image-SFT-MLX-compact"
REVISION = "310ab44798e3711eb134c1b7d30a73083c91af58"
PATTERNS = ["*.json", "*.jinja", "LICENSE", "NOTICE", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*",
            "vae/*"]
STUDENT = {"repo": "VisualInference/MOSS-Image-SFT-Turbo-MLX", "file": "turbo_student.safetensors",
           "revision": "4dd94eeb06457a12ad5ca8f3b2df5674d06378a7"}
DESTRIPE = {"repo": "nynxz/MOSS-Image", "file": "loras/moss_sft_destripe_top64_bf16.safetensors",
            "revision": "7b071d5915362f0101f0aa913315fd10a0772059", "strength": 1.5}
# sizes near one megapixel (the model's training size), multiples of 16
SIZES = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768), (768, 768)]


def installed():
    return PY.exists() and MARKER.exists() and MARKER.read_text().strip() == COMMIT and os.path.exists(os.path.realpath(PY))


def turbo_files(download=False):
    """Local paths of the turbo student and the destripe LoRA (None if not on this Mac and download is False)."""
    from huggingface_hub import hf_hub_download
    out = []
    for f in (STUDENT, DESTRIPE):
        try:
            out.append(hf_hub_download(f["repo"], f["file"], revision=f["revision"], local_files_only=not download))
        except Exception:
            if download:
                raise
            return None
    return out


def weights_dir(download=False):
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(REPO, revision=REVISION, allow_patterns=PATTERNS, local_files_only=not download)
    except Exception:
        if download:
            raise
        return None


class MossEngine(mlx_engine.MlxEngine):
    def __init__(self, emit, cancel_event, pause_event=None):
        super().__init__(emit, cancel_event, pause_event)
        self.python = PY
        self.worker_script = HERE / "moss_worker.py"

    def install(self):
        if not mlx_engine.supported():
            raise EngineError("MOSS-Image needs a Mac with Apple Silicon (M1 or newer).")
        if installed():
            return
        VENV.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(VENV.parent).free / 2**30
        if free < INSTALL_GB + 1:
            raise EngineError(f"Installing MOSS-Image's engine needs about {INSTALL_GB:.0f} GB free; only {free:.1f} GB is free.")
        self.emit({"type": "status", "msg": "Installing MOSS-Image's engine (one time, a few minutes)…"})
        if VENV.exists():
            shutil.rmtree(VENV, ignore_errors=True)
        base = getattr(sys, "_base_executable", None) or sys.executable
        self._run([base, "-m", "venv", str(VENV)])
        self._run([str(PY), "-m", "pip", "install", "--upgrade", "pip"], quiet=True)
        self._run([str(PY), "-m", "pip", "install", SPEC])
        MARKER.write_text(COMMIT)
        self.emit({"type": "status", "msg": "MOSS-Image's engine installed"})
