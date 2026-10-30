"""Embeddings (textual inversions): small files that teach the model one new word, such as EasyNegative or
a style or a face. Type the embedding's name in the prompt or negative prompt, or click it above the
negative prompt, and it's loaded for that picture.

Where they come from (Settings → Embeddings):
  * Protoart's own embeddings folder (~/.protoartstudio/embeddings): "Add embedding files…" copies files here,
    and embeddings from Civitai are downloaded here;
  * any folders you add (sub-folders included; read in place, nothing is copied);
  * embeddings sitting in your LoRA folders (.safetensors), which are no longer mistaken for LoRAs;
  * DiffusionBee's embeddings (.tdict, SD 1.5), while DiffusionBee is switched on.

The type is read from the file: SD 1.5 embeddings hold 768-wide vectors; SDXL ones (Pony, Illustrious and
NoobAI included) hold a 768-wide "clip_l" and a 1280-wide "clip_g" part. Only embeddings for the chosen
model's type are listed and used. .pt files are read with torch's weights-only loader (no code can run).
"""
import os
import re
import threading
from pathlib import Path

import nsfw
import paths

EXTRA_DIRS = []           # your embedding folders (Settings → Embeddings), set by the backend
EXTENSIONS = (".safetensors", ".pt", ".bin")
MAX_MB = 50               # embeddings are kilobytes to a few MB; anything bigger is something else
FAMILY_NAMES = {"sd15": "SD 1.5", "sdxl": "SDXL", "sd2": "SD 2"}
_cache = {}               # path -> (mtime, size, info or None)
_lock = threading.Lock()


# ------------------------------------------------------------------ what's in a file
def _shape(x):
    """A tensor's shape, from a real tensor or a safetensors header entry."""
    if isinstance(x, dict):
        return list(x.get("shape") or [])
    return list(getattr(x, "shape", []) or [])


def _width(shape):
    return int(shape[-1]) if shape else 0


def _vectors(shape):
    return int(shape[0]) if len(shape) == 2 else 1 if len(shape) == 1 else 0


def classify(entries):
    """entries: {key: tensor or safetensors header entry}. -> {"family", "vectors"} or None (not an embedding)."""
    keys = [k for k in entries if k != "__metadata__"]
    if not keys or len(keys) > 6:
        return None
    lowered = " ".join(keys).lower()
    if any(m in lowered for m in ("lora", "lokr", "hada", "unet", "down_blocks", "transformer")):
        return None
    if "clip_l" in entries and "clip_g" in entries:
        l, g = _shape(entries["clip_l"]), _shape(entries["clip_g"])
        if _width(l) == 768 and _width(g) == 1280 and len(l) <= 2 and len(g) <= 2:
            return {"family": "sdxl", "vectors": _vectors(l)}
        return None
    key = _sd_key(keys)
    if key is None:
        return None
    shape = _shape(entries[key])
    if len(shape) not in (1, 2) or _vectors(shape) > 75:
        return None
    width = _width(shape)
    if width == 768:
        return {"family": "sd15", "vectors": _vectors(shape)}
    if width == 1024:
        return {"family": "sd2", "vectors": _vectors(shape)}  # SD 2.x: listed, but can't be used
    return None


def _sd_key(keys):
    for k in keys:
        if k == "emb_params" or k.startswith("string_to_param"):
            return k
    return keys[0] if len(keys) == 1 else None


def _flatten_pt(data):
    """A1111 .pt layout -> {key: tensor}: {"string_to_param": {"*": t}, "name": …} or a plain dict of tensors."""
    out = {}
    if not isinstance(data, dict):
        return out
    stp = data.get("string_to_param")
    if isinstance(stp, dict):
        for k, v in stp.items():
            out["string_to_param." + str(k)] = v
        return out
    for k, v in data.items():
        if hasattr(v, "shape"):
            out[str(k)] = v
    return out


def _load_pt(path):
    import torch
    return _flatten_pt(torch.load(path, map_location="cpu", weights_only=True))


def inspect_file(path):
    """-> {"family", "vectors"} or None. Reads only the header of a .safetensors file."""
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext not in EXTENSIONS or os.path.getsize(p) > MAX_MB * 1e6:
        return None
    try:
        if ext == ".safetensors":
            from loras import read_safetensors_header
            return classify(read_safetensors_header(p))
        return classify(_load_pt(p))
    except Exception:
        return None


def is_embedding_file(path):
    return inspect_file(path) is not None


# ------------------------------------------------------------------ finding them
def _info(p, st):
    c = _cache.get(str(p))
    if c and c[0] == st.st_mtime and c[1] == st.st_size:
        return c[2]
    info = inspect_file(p)
    _cache[str(p)] = (st.st_mtime, st.st_size, info)
    return info


_NEG = None


def is_negative_name(stem):
    """Embeddings meant for the negative prompt (EasyNegative, bad-hands, ng_deepnegative, verybadimagenegative…), by
    name: everything else (your own trained characters and styles, too) goes in the prompt."""
    global _NEG
    import re
    if _NEG is None:
        _NEG = re.compile(r"neg|unaesthetic|verybad|worst|lowres|^ng[_-]|^bad|bad[-_ ]?(hand|prompt|artist|quality|dream|anatomy|picture)",
                          re.I)
    return bool(_NEG.search(str(stem)))


def scan(folder, safetensors_only=False):
    out = []
    d = Path(folder)
    if not d.is_dir():
        return out
    exts = (".safetensors",) if safetensors_only else EXTENSIONS
    for p in sorted(d.glob("**/*")):
        if p.name.startswith("._") or p.suffix.lower() not in exts:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size > MAX_MB * 1e6:
            continue
        info = _info(p, st)
        if info:
            row = {"name": p.stem, "file": str(p), "family": info["family"], "vectors": info["vectors"],
                   "size_kb": max(1, round(st.st_size / 1024)), "negative": is_negative_name(p.stem)}
            if nsfw.name_flag(p.stem):
                row["nsfw_auto"] = True
            out.append(row)
    return out


def own_dir():
    return str(paths.HOME / "embeddings")


def all_embeddings():
    """Every embedding found, one per name (your folders first, then LoRA folders, then DiffusionBee)."""
    import loras
    out, taken = [], set()

    def add(rows, source):
        for r in rows:
            if r["name"].lower() in taken:
                continue
            taken.add(r["name"].lower())
            out.append(dict(r, source=source))

    with _lock:
        add(scan(own_dir()), "protoart")
        for d in EXTRA_DIRS:
            try:
                add(scan(d), "folder")
            except Exception:
                pass
        for d in [str(paths.LORAS_DIR)] + list(loras.EXTRA_DIRS):
            try:
                add(scan(d, safetensors_only=True), "lora_folder")
            except Exception:
                pass
    try:
        import diffusionbee
        add([{"name": i["name"], "file": i["file"], "family": i.get("family") or "sd15", "vectors": None,
              "size_kb": int((i.get("size_mb") or 0) * 1024)} for i in diffusionbee.embeddings()], "diffusionbee")
    except Exception:
        pass
    return out


def for_family(family):
    return [e for e in all_embeddings() if e["family"] == family]


# ------------------------------------------------------------------ loading
def load_vectors(row):
    """-> {"clip_l": tensor} for SD 1.5, {"clip_l": t, "clip_g": t} for SDXL."""
    if row.get("source") == "diffusionbee":
        import tdict_reader
        emb = tdict_reader.TDict(row["file"]).embedding()
        t = next(iter(emb.values())) if isinstance(emb, dict) else emb
        return {"clip_l": t}
    p = row["file"]
    if p.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        data = load_file(p)
    else:
        data = _load_pt(p)
    if "clip_l" in data and "clip_g" in data:
        return {"clip_l": data["clip_l"], "clip_g": data["clip_g"]}
    key = _sd_key([k for k in data if k != "__metadata__"])
    if key is None:
        raise ValueError("not an embedding")
    t = data[key]
    return {"clip_l": t if t.ndim == 2 else t.unsqueeze(0)}


# ------------------------------------------------------------------ adding files
def safe_name(name):
    base, ext = os.path.splitext(os.path.basename(str(name)))
    base = re.sub(r'[\\/:*?"<>|\s]+', "_", base).strip("._") or "embedding"
    return base + ext.lower()


def add_files(files, dest=None):
    """Copy embedding files into Protoart's embeddings folder. Never overwrites; skips anything that isn't an
    embedding. -> {"added": [names], "skipped": [{"file", "why"}]}."""
    import shutil
    dest = dest or own_dir()
    os.makedirs(dest, exist_ok=True)
    added, skipped = [], []
    for f in files or []:
        f = str(f)
        if not os.path.isfile(f):
            skipped.append({"file": f, "why": "not found"})
            continue
        info = inspect_file(f)
        if not info:
            skipped.append({"file": os.path.basename(f), "why": "not an embedding (LoRAs go in a LoRA folder)"})
            continue
        target = os.path.join(dest, safe_name(f))
        if os.path.exists(target):
            skipped.append({"file": os.path.basename(f), "why": "one with this name is already there"})
            continue
        shutil.copy2(f, target)
        added.append(os.path.splitext(os.path.basename(target))[0])
    return {"added": added, "skipped": skipped, "folder": dest}
