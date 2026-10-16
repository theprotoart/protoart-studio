"""Add new base models from Hugging Face as they're released.

Paste a model id ("owner/name"), a link to it, or a link to a single .safetensors checkpoint.
We read only small files (model_index.json / the safetensors header) to work out what kind of
model it is and how big the download will be; the real download happens the first time you
generate with it.
"""
import json
import re

FAMILY_BY_PIPELINE = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "ErnieImagePipeline": "ernie",
}
FAMILY_LABEL = {"sd15": "SD 1.5", "sdxl": "SDXL", "ernie": "ERNIE", "other": "Other"}
DEFAULTS = {
    "sd15": {"width": 512, "height": 512, "steps": 25, "cfg": 7.0},
    "sdxl": {"width": 1024, "height": 1024, "steps": 30, "cfg": 6.0},
    "ernie": {"width": 1024, "height": 1024, "steps": 8, "cfg": 1.0},
    "other": {"width": 1024, "height": 1024, "steps": 28, "cfg": 4.5},
}

USER_MODELS = []  # set by the backend from the saved setting "user_models"


def parse_source(text):
    """-> (repo_id, filename or None)"""
    t = (text or "").strip()
    t = re.sub(r"^https?://(www\.)?huggingface\.co/", "", t)
    t = t.split("?")[0].strip("/")
    m = re.match(r"^([\w.-]+/[\w.-]+)/(?:blob|resolve)/[^/]+/(.+)$", t)
    if m:
        return m.group(1), m.group(2)
    parts = t.split("/")
    if len(parts) >= 3 and parts[-1].endswith((".safetensors", ".ckpt")):
        return "/".join(parts[:2]), "/".join(parts[2:])
    if len(parts) == 2 and all(re.match(r"^[\w.-]+$", p) for p in parts):
        return t, None
    raise ValueError("Paste a Hugging Face model id like 'owner/model-name', or a link to the model page.")


def model_id_for(repo, filename=None):
    return "hf:" + repo + (":" + filename if filename else "")


def _family_from_keys(keys):
    joined = "\n".join(keys)
    if "conditioner.embedders.1" in joined:
        return "sdxl"
    if "model.diffusion_model." in joined:
        return "sd15"
    return None


def inspect(text):
    from huggingface_hub import HfApi, hf_hub_download
    repo, filename = parse_source(text)
    api = HfApi()
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:
        msg = str(e)
        import net
        problem = net.hf_auth_problem(e, repo)
        if problem:
            raise ValueError(problem)
        if "404" in msg:
            raise ValueError(f"Couldn't find '{repo}' on Hugging Face.")
        raise ValueError(f"Couldn't reach Hugging Face: {msg[:200]}")
    files = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    notes = []
    if filename:
        if filename not in files:
            raise ValueError(f"'{filename}' isn't in {repo}.")
        family = None
        try:
            from huggingface_hub import get_safetensors_metadata
            meta = get_safetensors_metadata(repo, filename=filename)
            keys = [k for fm in meta.files_metadata.values() for k in fm.tensors]
            family = _family_from_keys(keys)
        except Exception:
            pass
        if family is None:
            raise ValueError("Only SD 1.5 and SDXL single-file checkpoints can be added this way. "
                             "For other models, use the model's main Hugging Face id.")
        size = files[filename]
        class_name = None
        variant = None
    else:
        if "model_index.json" not in files:
            raise ValueError(f"{repo} isn't a diffusers model (no model_index.json). If it's a single checkpoint, "
                             "paste the link to its .safetensors file instead.")
        with open(hf_hub_download(repo, "model_index.json")) as f:
            index = json.load(f)
        class_name = index.get("_class_name", "")
        family = FAMILY_BY_PIPELINE.get(class_name, "other")
        variant = "fp16" if any(".fp16." in n for n in files) else None
        comps = [k for k, v in index.items() if not k.startswith("_") and isinstance(v, list) and v[0]]
        size = 0
        for name, sz in files.items():
            top = name.split("/")[0]
            if top not in comps or not name.endswith((".safetensors", ".gguf")):
                continue
            if variant and ".fp16." not in name and any(n.startswith(top + "/") and ".fp16." in n for n in files):
                continue  # skip the full-precision copy when an fp16 one exists
            size += sz
        if family == "other":
            notes.append(f"{class_name}: runs with the model's own sampler; LoRAs and Fast mode aren't available.")
    gb = round(size / 1e9, 1)
    if gb > 12:
        notes.append("Large model: on a Mac with 8–16 GB it may be very slow or not fit.")
    if getattr(info, "gated", False):
        import os
        notes.append("This model is gated: open its page on huggingface.co while signed in and accept its licence"
                     + ("." if os.environ.get("HF_TOKEN") else ", and save a Hugging Face access token below (free account)."))
    label = (filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] if filename else repo.split("/")[1]).replace("_", " ")
    return {"id": model_id_for(repo, filename), "repo": repo, "file": filename, "family": family,
            "family_label": FAMILY_LABEL[family], "class_name": class_name, "variant": variant,
            "download_gb": gb, "label": label, "notes": notes}


def to_spec(m):
    """Saved entry -> engine model spec."""
    family = m["family"]
    gb = float(m.get("download_gb") or 0)
    spec = {
        "label": f"{m.get('label') or m['repo']} (added)", "family": family, "repo": m["repo"],
        "download_gb": gb, "min_ram_gb": max(8, round(gb * 1.3)), "quant": None, "source": "hub",
        "defaults": dict(m.get("defaults") or DEFAULTS[family]), "class_name": m.get("class_name"),
    }
    if m.get("file"):
        spec["hub_file"] = m["file"]
    if m.get("variant"):
        spec["variant"] = m["variant"]
    return spec


# ------------------------------------------------------------------ models from your own files (offline)
MODEL_FAMILY = {}   # saved setting "model_families": model id -> "sd15"/"sdxl" you picked for a model file
LOCAL_MODELS = []   # saved setting "local_models": files or diffusers folders you added
MODEL_DIRS = []     # saved setting "model_dirs": folders scanned for checkpoints
_scan_cache = {}
_dirs_result = {"key": None, "at": 0.0, "found": []}  # whole-scan result, reused for a few seconds
_folder_cache = {}


def invalidate():
    """Call after the watched folders or added models change."""
    _dirs_result["key"] = None


def _safetensors_keys(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 200_000_000:
            raise ValueError("not a safetensors file")
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"], header.get("__metadata__") or {}


def inspect_local(path):
    """Work out what a file or folder on this Mac is. No internet."""
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        idx = p / "model_index.json"
        if not idx.is_file():
            raise ValueError(f"{p.name} isn't a diffusers model folder (it has no model_index.json).")
        class_name = json.loads(idx.read_text()).get("_class_name", "")
        family = FAMILY_BY_PIPELINE.get(class_name, "other")
        ck = (str(p), idx.stat().st_mtime)
        size = _folder_cache.get(ck)
        if size is None:  # adding up a big folder is slow: remember it until model_index.json changes
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.suffix in (".safetensors", ".bin", ".gguf"))
            _folder_cache[ck] = size
        notes = [] if family != "other" else [f"{class_name}: runs with the model's own sampler; LoRAs and Fast mode aren't available."]
        return {"id": "local:" + str(p), "path": str(p), "kind": "folder", "family": family,
                "family_label": FAMILY_LABEL[family], "class_name": class_name, "label": p.name,
                "download_gb": round(size / 1e9, 1), "notes": notes}
    if not p.is_file():
        raise ValueError(f"Can't find {path}.")
    suffix = p.suffix.lower()
    notes = []
    if suffix == ".safetensors":
        keys, meta = _safetensors_keys(p)
        if any(m in k.lower() for k in keys for m in ("lora_up", "lora_down", "lora_a", "lora_b")):
            raise ValueError(f"{p.name} is a LoRA, not a base model. Add its folder under LoRA folders instead.")
        family = _family_from_keys(keys)
        if family is None:
            raise ValueError(f"{p.name}: only SD 1.5 and SDXL single-file checkpoints can be added as files. "
                             "For other model types, add the model's folder (the one with model_index.json).")
    elif suffix == ".ckpt":
        family = "sdxl" if p.stat().st_size > 5e9 else "sd15"
        notes.append("Older .ckpt format: guessed the type from its size. .safetensors files are safer and load faster.")
    else:
        raise ValueError("Pick a .safetensors or .ckpt checkpoint, or a diffusers model folder.")
    return {"id": "local:" + str(p), "path": str(p), "kind": "file", "family": family,
            "family_label": FAMILY_LABEL[family], "class_name": None,
            "label": p.stem.replace("_", " "), "download_gb": round(p.stat().st_size / 1e9, 1), "notes": notes}


def inspect_any(path):
    """Like inspect_local, but a plain folder (not a diffusers model) is searched for models inside it.
    -> (found list, notes list). Raises ValueError with a friendly message when nothing usable is there."""
    from pathlib import Path
    p = Path(path)
    if not p.is_dir() or (p / "model_index.json").is_file():
        return [inspect_local(p)], []
    found, notes, seen = [], [], set()
    for idx in list(p.glob("*/model_index.json")) + list(p.glob("*/*/model_index.json")):
        try:
            found.append(inspect_local(idx.parent))
            seen.add(idx.parent)
        except Exception:
            pass
    skipped_small = 0
    for f in sorted(list(p.rglob("*.safetensors")) + list(p.rglob("*.ckpt"))):
        if f.name.startswith("._") or any(parent in seen for parent in f.parents):
            continue
        try:
            if f.stat().st_size < 1.5e9:
                skipped_small += 1  # LoRAs, embeddings, VAEs: not base models
                continue
            found.append(inspect_local(f))
        except Exception:
            pass
    tdicts = [f for f in p.rglob("*.tdict")]
    if tdicts:
        notes.append(f"{len(tdicts)} DiffusionBee file(s) (.tdict) in {p.name}: add this folder under Settings → "
                     "LoRA folders (or watch it as a models folder) and they show up: LoRAs in the LoRA list, "
                     "models as “(DiffusionBee)”.")
    if skipped_small and not found:
        notes.append(f"{skipped_small} smaller file(s) look like LoRAs or embeddings: add their folder under LoRA folders instead.")
    if not found:
        if notes:
            raise ValueError(" ".join(notes))
        raise ValueError(f"No base models found in {p.name} (looked for .safetensors / .ckpt checkpoints over 1.5 GB "
                         "and diffusers model folders).")
    return found, notes


def scan_model_dirs():
    """Checkpoints (.safetensors over 1.5 GB that aren't LoRAs) and diffusers folders inside MODEL_DIRS.
    The model list is asked for often, so a scan is reused for 10 seconds (walking big folders is slow)."""
    import time
    key = tuple(MODEL_DIRS)
    if _dirs_result["key"] == key and time.time() - _dirs_result["at"] < 10:
        return list(_dirs_result["found"])
    found = _scan_dirs()
    _dirs_result.update(key=key, at=time.time(), found=found)
    return list(found)


def _scan_dirs():
    from pathlib import Path
    found = []
    for d in MODEL_DIRS:
        root = Path(d)
        if not root.is_dir():
            continue
        for idx in list(root.glob("*/model_index.json")) + ([root / "model_index.json"] if (root / "model_index.json").exists() else []):
            try:
                found.append(inspect_local(idx.parent))
            except Exception:
                pass
        for f in root.rglob("*.safetensors"):
            if f.name.startswith("._") or "model_index.json" in [x.name for x in f.parent.parent.glob("model_index.json")]:
                continue
            try:
                st = f.stat()
                if st.st_size < 1.5e9:
                    continue
                key = (str(f), st.st_mtime, st.st_size)
                if key not in _scan_cache:
                    try:
                        _scan_cache[key] = inspect_local(f)
                    except Exception:
                        _scan_cache[key] = None
                if _scan_cache[key]:
                    found.append(dict(_scan_cache[key], source_dir=str(root)))
            except OSError:
                continue
    return found


def local_to_spec(m):
    family = m["family"]
    spec = {
        "label": f"{m.get('label') or m['path']} (your file)", "family": family, "download_gb": 0,
        "min_ram_gb": max(8, round(float(m.get("download_gb") or 0) * 1.3)), "quant": None, "source": "local",
        "defaults": dict(DEFAULTS[family]), "class_name": m.get("class_name"),
    }
    if m.get("kind") == "folder":
        spec["local_dir"] = m["path"]
    else:
        spec["file"] = m["path"]
        spec["low_memory_min_ram_gb"] = 8 if family == "sdxl" else None
    return spec
