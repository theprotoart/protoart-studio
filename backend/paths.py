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
EMBEDDINGS_DIR = HOME / "embeddings"  # embeddings (textual inversions) added in the app
CONVERTED_DIR = HOME / "converted"  # models the app converted (Models → Convert)

# 📦 Model files the app keeps (Settings → Model files: move them to another drive). name -> attribute here
STORE = {"loras": "LORAS_DIR", "models": "MODELS_DIR", "quantized": "QUANT_DIR", "vaes": "VAE_DIR",
         "controlnets": "CONTROLNET_DIR", "embeddings": "EMBEDDINGS_DIR", "converted": "CONVERTED_DIR"}
STORE_ROOT = None  # the folder they're in when moved off the data folder (None: in the data folder)


def store_default(name):
    return HOME / name


def apply_store(root):
    """Use <root>/<name> for each kind of model file (a moved store), or the data folder when root is None."""
    global STORE_ROOT
    STORE_ROOT = Path(root) if root else None
    for name, attr in STORE.items():
        globals()[attr] = (STORE_ROOT / name) if STORE_ROOT else store_default(name)
    for name in ("loras", "models", "quantized", "vaes", "controlnets"):
        try:
            globals()[STORE[name]].mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def ensure_dirs():
    """Make the app's folders. One that can't be made (a Finder alias or a file in its place, a link to a drive that
    isn't plugged in) no longer stops the app from starting: the images go to "images (Protoart)" next to it instead,
    and the other folders are skipped and reported. (To keep images elsewhere, use Settings → Image folder.)"""
    global IMAGES_DIR
    HOME.mkdir(parents=True, exist_ok=True)
    for d in (IMAGES_DIR, LORAS_DIR, MODELS_DIR, WEIGHTS_DIR, QUANT_DIR, VAE_DIR, CONTROLNET_DIR):  # (embeddings, converted: made when used)
        try:
            d.mkdir(parents=True, exist_ok=True)
            if not d.is_dir():
                raise NotADirectoryError(str(d))
        except OSError as e:
            print(f"[paths] can't use {d} as a folder ({e}); is it a Finder alias or an unplugged drive?", flush=True)
            if d == IMAGES_DIR:
                IMAGES_DIR = HOME / "images (Protoart)"
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)


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
