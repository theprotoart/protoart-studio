"""Image → prompt with a modern vision model: Qwen3.5 4B (Apache-2.0, image + text) on the Apple GPU through
mlx-vlm. It looks at the picture and writes a detailed, natural-language prompt (subject, clothing, pose, setting,
light, colours, framing, style): what FLUX, FLUX.2 and ERNIE read best.

It runs in its own Python environment (~/.protoartstudio/venv-vlm, installed once) so its libraries can never
disturb the fast engine's, and as a short-lived process (vlm_worker.py) that exits after each picture, so its
~3 GB of memory is given back straight away (it matters on 8 GB Macs).
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_HOME = Path(os.environ.get("PROTOART_HOME") or (Path.home() / ".protoartstudio"))
VENV = _HOME / "venv-vlm"
PY = VENV / "bin" / "python"
MARKER = VENV / ".protoart-installed"
SPEC = "mlx-vlm==0.7.4"   # exact tested version
INSTALL_GB = 1.5

REPO = "mlx-community/Qwen3.5-4B-MLX-4bit"
REVISION = "32f3e8ecf65426fc3306969496342d504bfa13f3"   # pinned: the files can't change under us
DOWNLOAD_GB = 3.1
LABEL = "Qwen3.5 4B"

STYLES = {
    # for FLUX / FLUX.2 / ERNIE / SDXL: one rich paragraph
    "natural": ("Describe this image as a prompt for an AI image generator. Write one detailed paragraph of plain "
                "sentences covering: the main subject (who or what, appearance, clothing, pose, expression), what "
                "they are doing, the setting and background, the lighting, the main colours, the camera angle and "
                "framing, and the art style or medium (photo, painting, anime, 3D render…). Mention any visible text "
                "in quotes. Do not start with 'This image' or 'The image'. No lists, no headings, no preamble."),
    # shorter, for quick ideas
    "short": ("Describe this image as a short prompt for an AI image generator: one or two sentences with the "
              "subject, setting, lighting and style. Do not start with 'This image'. No preamble."),
}


class VlmError(Exception):
    pass


class Cancelled(Exception):
    pass


def supported():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def installed():
    return PY.exists() and MARKER.exists() and os.path.exists(os.path.realpath(PY))


def downloaded():
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_files_only=True)
        return True
    except Exception:
        return False


def _run(cmd, emit, cancel_event, what):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(HERE))
    tail, last = [], 0
    for line in p.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        print("[vlm-install]", line, file=sys.stderr, flush=True)
        if cancel_event is not None and cancel_event.is_set():
            p.kill()
            raise Cancelled()
        if line.startswith(("Collecting", "Downloading", "Installing")) and time.time() - last > 1:
            emit({"type": "status", "msg": f"{what}: " + line[:90]})
            last = time.time()
    if p.wait() != 0:
        raise VlmError(f"{what} failed:\n" + "\n".join(tail[-8:]))


def install(emit, cancel_event=None):
    """One-time install of mlx-vlm into its own environment (needs the internet once)."""
    if not supported():
        raise VlmError("Describing with Qwen3.5 needs a Mac with Apple Silicon (M1 or newer).")
    if installed():
        return
    if not allow_install_now():
        raise VlmError("offline: the picture reader isn't installed yet")  # retried online by the backend
    VENV.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(VENV.parent).free / 2**30
    if free < INSTALL_GB + DOWNLOAD_GB + 1:
        raise VlmError(f"Setting up the picture reader needs about {INSTALL_GB + DOWNLOAD_GB:.0f} GB free; "
                       f"only {free:.1f} GB is free.")
    emit({"type": "status", "msg": "Setting up the picture reader (one time, a few minutes)…"})
    if VENV.exists():  # half-finished install: start over
        shutil.rmtree(VENV, ignore_errors=True)
    base = getattr(sys, "_base_executable", None) or sys.executable
    _run([base, "-m", "venv", str(VENV)], emit, cancel_event, "Setting up the picture reader")
    _run([str(PY), "-m", "pip", "install", "--upgrade", "pip"], emit, cancel_event, "Setting up the picture reader")
    _run([str(PY), "-m", "pip", "install", SPEC], emit, cancel_event, "Installing the picture reader")
    MARKER.write_text(SPEC)


def allow_install_now():
    try:
        import net
        return net.is_online()
    except Exception:
        return True


def ready():
    """Installed and downloaded already (text jobs use it only then: they never start a 3 GB download)."""
    return supported() and installed() and downloaded()


def chat(text, emit=None, cancel_event=None, max_tokens=500):
    """A text-only answer from the same model (no picture), as written: line breaks kept, thinking removed.
    Only when ready(); runs the worker once, like describe()."""
    if not ready():
        raise VlmError(f"{LABEL} isn't set up")
    raw = _ask({"repo": REPO, "revision": REVISION, "image": None, "max_tokens": int(max_tokens), "prompt": text},
               emit or (lambda m: None), cancel_event, dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1"))
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    return raw.replace("<think>", "").strip()


def describe(image_path, style="natural", emit=None, cancel_event=None, allow_download=True, max_tokens=320):
    """Returns the prompt text. Runs vlm_worker.py once; it exits afterwards (memory back at once)."""
    emit = emit or (lambda m: None)
    install(emit, cancel_event)
    if not downloaded():
        if not allow_download:
            raise VlmError(f"offline: {LABEL} isn't downloaded yet")
        emit({"type": "status", "msg": f"Downloading {LABEL} (~{DOWNLOAD_GB:.0f} GB, first time only)…"})
    else:
        emit({"type": "status", "msg": f"Reading the picture with {LABEL}…"})
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if allow_download:
        env["HF_HUB_OFFLINE"] = "0"
    req = {"repo": REPO, "revision": REVISION, "image": str(image_path), "max_tokens": int(max_tokens),
           "prompt": STYLES.get(style, STYLES["natural"])}
    return clean(_ask(req, emit, cancel_event, env))


def _ask(req, emit, cancel_event, env):
    p = subprocess.Popen([str(PY), str(HERE / "vlm_worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=str(HERE), env=env)
    err_tail = []

    def read_err():
        for line in p.stderr:
            err_tail.append(line.rstrip())
            del err_tail[:-40]
            print("[vlm]", line.rstrip(), file=sys.stderr, flush=True)

    threading.Thread(target=read_err, daemon=True).start()
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.close()
    result = None
    for line in p.stdout:
        if cancel_event is not None and cancel_event.is_set():
            p.kill()
            raise Cancelled()
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("type") == "status":
            emit({"type": "status", "msg": msg.get("msg", "")})
        elif msg.get("type") in ("done", "error"):
            result = msg
    p.wait()
    if result is None:
        raise VlmError("The picture reader stopped unexpectedly:\n" + "\n".join(err_tail[-6:]))
    if result["type"] == "error":
        raise VlmError(result.get("msg") or "The picture reader failed.")
    return result.get("text") or ""


def clean(text):
    """Tidy the model's answer into a prompt: no thinking block, quotes, preamble or line breaks."""
    t = text
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    t = t.replace("<think>", "").strip().strip('"').strip()
    for pre in ("Prompt:", "Here is a prompt:", "Here's a prompt:", "Sure!", "Sure,"):
        if t.lower().startswith(pre.lower()):
            t = t[len(pre):].strip()
    return " ".join(t.split())
