"""Use the models, LoRAs and embeddings you already have in DiffusionBee (~/.diffusionbee).

Nothing is copied or converted: .tdict files are read in place. A small index of what each
file contains is kept in ~/.protoartstudio/diffusionbee_index.json so startup stays fast.
"""
import json
import os
import threading
from pathlib import Path

import paths
import tdict_reader

ROOT = Path(os.environ.get("DIFFUSIONBEE_HOME") or (Path.home() / ".diffusionbee"))
SUBDIRS = ("imported_models", "downloaded_assets")

enabled = True  # set from Settings (stored as app setting "import_diffusionbee")
_lock = threading.Lock()
_loaded_index = False


def dirs():
    return [ROOT / s for s in SUBDIRS]


def available():
    return any(d.is_dir() for d in dirs())


def _index_path():
    return paths.HOME / "diffusionbee_index.json"


def _load_index():
    global _loaded_index
    if _loaded_index:
        return
    _loaded_index = True
    try:
        data = json.loads(_index_path().read_text())
        for p, (mtime, size, info) in data.items():
            tdict_reader._cache[p] = (mtime, size, info)
    except Exception:
        pass


def _save_index():
    try:
        tmp = _index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps({p: list(v) for p, v in tdict_reader._cache.items()}))
        tmp.replace(_index_path())
    except Exception:
        pass


def extra_dirs():
    """Your own folders that can hold DiffusionBee .tdict files too (copies kept outside DiffusionBee):
    the LoRA folders and watched model folders from Settings, and Protoart's own LoRA/model folders."""
    import loras
    import model_hub
    out = []
    for d in [str(paths.LORAS_DIR), str(paths.MODELS_DIR)] + list(loras.EXTRA_DIRS) + list(model_hub.MODEL_DIRS):
        if d and d not in out:
            out.append(d)
    return out


def items(force=False):
    """Everything found (supported or not). DiffusionBee's own folder only while switched on in Settings;
    .tdict files in your own folders always."""
    with _lock:
        _load_index()
        before = len(tdict_reader._cache)
        found = tdict_reader.scan(dirs()) if (enabled or force) else []
        seen = {os.path.realpath(i["file"]) for i in found}
        for i in tdict_reader.scan(extra_dirs(), recursive=True):
            if os.path.realpath(i["file"]) not in seen:
                seen.add(os.path.realpath(i["file"]))
                found.append(i)
        if len(tdict_reader._cache) != before:
            _save_index()
    return found


def allowed_roots():
    """Folders whose .tdict files may be converted (Tools → Convert DiffusionBee files)."""
    return [os.path.realpath(str(ROOT))] + [os.path.realpath(d) for d in extra_dirs()]


def checkpoints():
    return [i for i in items() if i["kind"] == "checkpoint" and i["supported"]]


def loras():
    return [i for i in items() if i["kind"] == "lora" and i["supported"]]


def embeddings():
    return [i for i in items() if i["kind"] == "embedding" and i["supported"]]


def summary():
    all_items = items(force=True)
    count = lambda kind, ok=True: sum(1 for i in all_items if i["kind"] == kind and i["supported"] == ok)
    skipped = [i for i in all_items if not i["supported"]]
    return {
        "enabled": enabled, "available": available(), "folder": str(ROOT),
        "checkpoints": count("checkpoint"), "loras": count("lora"), "embeddings": count("embedding"),
        "skipped": len(skipped),
        "skipped_kinds": sorted({i["kind"] + (f" ({i['family']})" if i["family"] not in ("unknown", None) else "")
                                 for i in skipped}),
    }


_orig_cache = {"mtime": None, "map": {}}


def originals():
    """tdict path -> the file it was imported from (DiffusionBee remembers this in
    locally_loaded_assets.json). The original may hold trigger words in its metadata."""
    f = ROOT / "locally_loaded_assets.json"
    try:
        mtime = f.stat().st_mtime
    except OSError:
        return {}
    if _orig_cache["mtime"] != mtime:
        m = {}
        try:
            for rec in json.loads(f.read_text(errors="ignore")).values():
                if isinstance(rec, dict) and rec.get("asset_path") and rec.get("asset_path_raw"):
                    m[rec["asset_path"]] = rec["asset_path_raw"]
        except Exception:
            pass
        _orig_cache.update(mtime=mtime, map=m)
    return _orig_cache["map"]
