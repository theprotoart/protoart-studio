"""Find model libraries other apps already keep on this Mac (Stability Matrix, ComfyUI, Automatic1111 / Forge,
InvokeAI, Fooocus) so Protoart Studio can use them where they are.

Read-only by design: this module only lists folders and reads file sizes and small settings files. It never
moves, copies, renames, deletes or writes anything inside another app's folders. Using a library just adds its
folders to the lists Protoart Studio looks in (the same as adding them by hand in Settings / Models).
"""
import json
import os
import re

MODEL_EXTS = (".safetensors", ".ckpt", ".gguf", ".sft", ".pt", ".pth", ".bin")
KINDS = ("checkpoints", "diffusion", "loras", "embeddings", "vaes", "controlnets")
KIND_LABEL = {"checkpoints": "checkpoints", "diffusion": "diffusion models", "loras": "LoRAs",
              "embeddings": "embeddings", "vaes": "VAEs", "controlnets": "ControlNets"}

# Where each app keeps each kind of file, relative to its library folder.
LAYOUTS = {
    "stability_matrix": {
        "name": "Stability Matrix",
        "checkpoints": ["Models/StableDiffusion"],
        "diffusion": ["Models/DiffusionModels", "Models/Unet"],
        "loras": ["Models/Lora", "Models/LyCORIS"],
        "embeddings": ["Models/TextualInversion", "Models/Embeddings"],
        "vaes": ["Models/VAE"],
        "controlnets": ["Models/ControlNet"],
    },
    "comfyui": {
        "name": "ComfyUI",
        "checkpoints": ["models/checkpoints"],
        "diffusion": ["models/diffusion_models", "models/unet"],
        "loras": ["models/loras"],
        "embeddings": ["models/embeddings"],
        "vaes": ["models/vae"],
        "controlnets": ["models/controlnet"],
    },
    "a1111": {
        "name": "Automatic1111 / Forge",
        "checkpoints": ["models/Stable-diffusion"],
        "loras": ["models/Lora", "models/LyCORIS"],
        "embeddings": ["embeddings"],
        "vaes": ["models/VAE"],
        "controlnets": ["models/ControlNet", "extensions/sd-webui-controlnet/models"],
    },
    "fooocus": {
        "name": "Fooocus",
        "checkpoints": ["models/checkpoints"],
        "loras": ["models/loras"],
        "embeddings": ["models/embeddings"],
        "vaes": ["models/vae"],
        "controlnets": ["models/controlnet"],
    },
    "invokeai": {
        "name": "InvokeAI",
        "checkpoints": ["models"],   # InvokeAI sorts by type inside; Protoart reads each file's header instead
    },
}

# What tells the layouts apart, when a folder could be more than one (ComfyUI and Fooocus share names).
_MARKERS = {
    "stability_matrix": ["Models/StableDiffusion", "Models/Lora", "Packages"],
    "comfyui": ["models/checkpoints", "comfy", "custom_nodes", "main.py"],
    "a1111": ["models/Stable-diffusion", "webui.py", "modules"],
    "fooocus": ["entry_with_update.py", "fooocus_version.py"],
    "invokeai": ["invokeai.yaml", "databases"],
}

MAX_WALK = 20000   # stop counting after this many entries, so a huge folder can't freeze the scan


def home():
    return os.path.expanduser("~")


def _isdir(p):
    try:
        return os.path.isdir(p)
    except OSError:
        return False


def _read_json(p):
    try:
        if os.path.getsize(p) > 2_000_000:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _paths_in(obj, keys=("librarypath", "datadirectory", "modelsdirectory", "modeldirectory", "basepath")):
    """String values under keys that name a folder, anywhere in a small settings file."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.replace("_", "").lower() in keys and v.strip():
                out.append(os.path.expanduser(v.strip()))
            elif isinstance(v, (dict, list)):
                out += _paths_in(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            out += _paths_in(v, keys)
    return out


def candidates(h=None):
    """(layout, folder) pairs worth checking. Nothing here is opened for writing."""
    h = h or home()
    sup = os.path.join(h, "Library", "Application Support")
    out = []

    # Stability Matrix: its own settings name the library folder; otherwise the usual places.
    sm_app = os.path.join(sup, "StabilityMatrix")
    for name in ("library.json", "settings.json"):
        for p in _paths_in(_read_json(os.path.join(sm_app, name))):
            out.append(("stability_matrix", p))
            out.append(("stability_matrix", os.path.join(p, "Data")))
    for base in (sm_app, os.path.join(h, "StabilityMatrix"), os.path.join(h, "Documents", "StabilityMatrix"),
                 os.path.join(h, "Applications", "StabilityMatrix"), "/Applications/StabilityMatrix",
                 os.path.join(h, "Desktop", "StabilityMatrix"), os.path.join(h, "Downloads", "StabilityMatrix")):
        out.append(("stability_matrix", base))
        out.append(("stability_matrix", os.path.join(base, "Data")))
    # Portable installs: a "Data" folder next to the app, wherever it was unzipped.
    for parent in (os.path.join(h, "Applications"), "/Applications", os.path.join(h, "Downloads"),
                   os.path.join(h, "Desktop"), os.path.join(h, "Documents"), h):
        try:
            for name in os.listdir(parent):
                if name.lower().startswith("stabilitymatrix") or name.lower().startswith("stability matrix"):
                    out.append(("stability_matrix", os.path.join(parent, name, "Data")))
                    out.append(("stability_matrix", os.path.join(parent, name)))
        except OSError:
            pass

    # ComfyUI (the desktop app keeps a base path in its config, and a yaml of extra model folders).
    comfy_cfg = os.path.join(sup, "ComfyUI")
    for p in _paths_in(_read_json(os.path.join(comfy_cfg, "config.json"))):
        out.append(("comfyui", p))
    try:
        with open(os.path.join(comfy_cfg, "extra_models_config.yaml"), "r", encoding="utf-8") as f:
            for m in re.finditer(r"base_path:\s*['\"]?([^'\"\n#]+)", f.read(200_000)):
                out.append(("comfyui", os.path.expanduser(m.group(1).strip())))
    except OSError:
        pass
    for base in ("ComfyUI", "Documents/ComfyUI", "Desktop/ComfyUI", "AI/ComfyUI", "Projects/ComfyUI", "comfyui",
                 "pinokio/api/comfy.git/app"):
        out.append(("comfyui", os.path.join(h, base)))

    for base in ("stable-diffusion-webui", "stable-diffusion-webui-forge", "stable-diffusion-webui-reForge",
                 "Documents/stable-diffusion-webui", "Documents/stable-diffusion-webui-forge",
                 "AI/stable-diffusion-webui", "AI/stable-diffusion-webui-forge", "sd-webui-forge"):
        out.append(("a1111", os.path.join(h, base)))
    for base in ("Fooocus", "Documents/Fooocus", "AI/Fooocus"):
        out.append(("fooocus", os.path.join(h, base)))
    for base in [os.environ.get("INVOKEAI_ROOT") or "", os.path.join(h, "invokeai"),
                 os.path.join(h, "Documents", "invokeai"), os.path.join(h, "InvokeAI")]:
        if base:
            out.append(("invokeai", base))
    return out


def _count(folder, limit=MAX_WALK):
    """How many model files are in a folder (and below), and how many gigabytes. Only sizes are read."""
    n, size, seen = 0, 0, 0
    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            seen += 1
            if f.startswith("._") or not f.lower().endswith(MODEL_EXTS):
                continue
            try:
                size += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
            n += 1
        if seen > limit:
            break
    return n, round(size / 1e9, 1)


def _guess_layout(folder):
    """The layout a folder you picked yourself follows, or None. A folder that IS a "Models"/"models" folder
    is turned into its parent."""
    folder = folder.rstrip("/") or "/"
    if os.path.basename(folder) in ("Models", "models") and _isdir(os.path.dirname(folder)):
        up = _guess_layout(os.path.dirname(folder))
        if up:
            return up
    scores = {}
    for lay, marks in _MARKERS.items():
        scores[lay] = sum(1 for m in marks if os.path.exists(os.path.join(folder, m)))
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        # No markers: any layout whose category folders exist will do (e.g. a copied "models" tree).
        for lay, spec in LAYOUTS.items():
            if any(_isdir(os.path.join(folder, rel)) for k in KINDS for rel in spec.get(k, [])) and lay != "invokeai":
                return lay, folder
        return None
    return best, folder


def inspect(layout, folder, count=True):
    """What a library holds, or None when it isn't one (no category folders). Read-only."""
    spec = LAYOUTS.get(layout)
    if not spec or not _isdir(folder):
        return None
    dirs, counts, total, gb = {}, {}, 0, 0.0
    for kind in KINDS:
        for rel in spec.get(kind, []):
            p = os.path.join(folder, rel)
            if not _isdir(p):
                continue
            dirs.setdefault(kind, []).append(p)
            if count:
                n, g = _count(p)
                counts[kind] = counts.get(kind, 0) + n
                total += n
                gb += g
    if not dirs:
        return None
    if layout == "invokeai" and not any(os.path.exists(os.path.join(folder, m)) for m in _MARKERS["invokeai"]):
        return None   # a plain "models" folder in some other place isn't an InvokeAI install
    return {"root": os.path.realpath(folder), "layout": layout, "name": spec["name"], "dirs": dirs,
            "counts": counts, "files": total, "gb": round(gb, 1)}


def detect(h=None, count=True):
    """Every model library found on this Mac, biggest first. Folders that resolve to the same place (a symlink,
    Stability Matrix's linked package folders) are listed once."""
    seen, out = set(), []
    for layout, folder in candidates(h):
        try:
            real = os.path.realpath(folder)
            if not _isdir(real):
                continue
            st = os.stat(real)
            key = (st.st_dev, st.st_ino)  # macOS ignores case: ~/ComfyUI and ~/comfyui are one folder
        except OSError:
            continue
        if key in seen:
            continue
        info = inspect(layout, real, count=count)
        if info:
            seen.add(key)
            out.append(info)
    out.sort(key=lambda x: -x["files"])
    return out


def inspect_any(folder):
    """A folder you picked in Settings: which app's layout it follows, and what's in it."""
    folder = os.path.expanduser(str(folder or ""))
    if not os.path.isabs(folder) or not _isdir(folder):
        raise ValueError("That folder isn't there any more.")
    g = _guess_layout(folder)
    if not g:
        raise ValueError("That folder doesn't look like a model library (no Models / checkpoints / Lora folders). "
                         "To use a plain folder of models, add it in Models → On this Mac → Watch a folder.")
    info = inspect(*g)
    if not info:
        raise ValueError("No model folders found in there.")
    return info


def summary_line(info):
    parts = [f"{info['counts'][k]} {KIND_LABEL[k]}" for k in KINDS if info.get("counts", {}).get(k)]
    return ", ".join(parts) if parts else "no model files yet"
