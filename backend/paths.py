"""Filesystem locations used by Protoart Studio.

Everything the app creates lives under ~/.protoartstudio unless PROTOART_HOME is set.
(Older installs used ~/.beestudio; that folder is still used if it hasn't been moved yet.)
Downloaded models use the normal Hugging Face cache (~/.cache/huggingface/hub), unless you pick another
folder for them in Settings (for example on an external drive): see apply_hub_cache().
"""
import json
import os
from pathlib import Path

LEGACY_HOME = Path.home() / ".beestudio"
_NEW_HOME = Path.home() / ".protoartstudio"
_env = os.environ.get("PROTOART_HOME") or os.environ.get("BEESTUDIO_HOME")
if _env:
    HOME = Path(_env)
elif not _NEW_HOME.exists() and LEGACY_HOME.exists():
    HOME = LEGACY_HOME  # not migrated yet
else:
    HOME = _NEW_HOME
IMAGES_DIR = HOME / "images"
LORAS_DIR = HOME / "loras"
MODELS_DIR = HOME / "models"  # user-supplied .safetensors checkpoints
DB_PATH = HOME / "library.sqlite"
WEIGHTS_DIR = HOME / "weights"  # RestoreFormer + face detector weights
QUANT_DIR = HOME / "quantized"  # 8-bit copies Protoart Studio makes itself (e.g. SDXL)
VAE_DIR = HOME / "vaes"  # VAE files (from Civitai or your own) for SD 1.5 / SDXL
CONTROLNET_DIR = HOME / "controlnets"  # your own SD 1.5 ControlNet files (e.g. QR Code Monster from Civitai)


def ensure_dirs():
    for d in (HOME, IMAGES_DIR, LORAS_DIR, MODELS_DIR, WEIGHTS_DIR, QUANT_DIR, VAE_DIR, CONTROLNET_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ where downloaded models are kept
DOWNLOADS_FILE = HOME / "downloads_location.json"
DEFAULT_HUB_CACHE = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def configured_hub_cache():
    """The folder chosen in Settings for downloaded models, or None for the default."""
    try:
        p = json.loads(DOWNLOADS_FILE.read_text()).get("path")
        return Path(p) if p else None
    except (OSError, ValueError):
        return None


def save_hub_cache(path):
    DOWNLOADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_FILE.write_text(json.dumps({"path": str(path) if path else None}))


HUB_CACHE_MISSING = None  # the chosen folder, if it couldn't be found at start-up (drive not plugged in)


def apply_hub_cache():
    """Called once at start-up, before any model library loads: point the download cache at the chosen folder.
    If that folder isn't there (an external drive that isn't connected), the default is used for this session
    and nothing is created on the missing drive."""
    global HUB_CACHE_MISSING
    chosen = configured_hub_cache()
    if not chosen:
        return
    if chosen.is_dir():
        os.environ["HF_HUB_CACHE"] = str(chosen)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(chosen)
    else:
        HUB_CACHE_MISSING = str(chosen)


def hub_cache():
    """Where downloaded models are right now."""
    return Path(os.environ.get("HF_HUB_CACHE") or DEFAULT_HUB_CACHE)


def folder_access(path):
    """"ok", "missing" (not there: renamed, or the drive isn't connected) or "denied" (macOS hasn't let this app read
    it yet: folders on external drives need the "removable volumes" permission the first time)."""
    try:
        if not os.path.isdir(path):
            return "missing"
        os.listdir(path)
        return "ok"
    except PermissionError:
        return "denied"
    except OSError:
        return "missing"
