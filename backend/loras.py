"""Find LoRA files and work out which model family each one belongs to.

Reads only the small JSON header of each .safetensors file (no torch needed),
so scanning a folder of hundreds of LoRAs is instant.
"""
import json
import re
import struct
from pathlib import Path

import nsfw

FAMILIES = ("sd15", "sdxl", "ernie", "flux")


# SDXL "flavours": fine-tuned SDXL families that are used differently from plain SDXL
# (own quality tags; their LoRAs often don't mix well with other SDXL models).
FLAVORS = ("pony", "illustrious", "noobai")
_FLAVOR_HINTS = {
    "noobai": ("noobai", "noob ai", "noob xl", "noobxl", "nai xl", "noob"),
    "illustrious": ("illustrious", "illustriousxl", "ilxl", "il xl", "waiillustrious", "wai illustrious", "wainsfw"),
    "pony": ("pony", "ponyxl", "ponydiffusion", "pdxl", "autismmix", "autism mix", "score 9"),
}


def detect_flavor(*texts):
    """'pony', 'illustrious', 'noobai' or None, from names and metadata (file name, model name, base model)."""
    import re as _re
    text = " ".join(str(t) for t in texts if t).lower()
    text = _re.sub(r"[_\-.]+", " ", text)
    squashed = text.replace(" ", "")
    for flavor in ("noobai", "illustrious", "pony"):  # NoobAI is built on Illustrious: check it first
        for h in _FLAVOR_HINTS[flavor]:
            if (" " in h and h in text) or h in text.split() or (len(h) > 4 and h.replace(" ", "") in squashed):
                return flavor
    return None


def flavor_from_meta(meta, *names):
    meta = meta or {}
    return detect_flavor(*names, meta.get("ss_sd_model_name"), meta.get("ss_base_model_version"),
                         meta.get("modelspec.title"), meta.get("modelspec.architecture"), meta.get("ss_output_name"))


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100_000_000:
            raise ValueError("not a safetensors file")
        return json.loads(f.read(n))


def detect_family(header):
    """Return 'sd15', 'sdxl', 'ernie' or 'unknown' for a safetensors header dict."""
    meta = header.get("__metadata__") or {}
    explicit = (meta.get("beestudio_family") or "").lower()
    if explicit in FAMILIES:
        return explicit
    base = (meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or "").lower()
    if "xl" in base:
        return "sdxl"
    if base.startswith("sd_v1") or "v1" in base and "stable-diffusion" in base:
        return "sd15"
    if "ernie" in base:
        return "ernie"
    if "flux" in base:
        return "flux"

    keys = [k for k in header if k != "__metadata__"]
    joined = "\n".join(keys)
    # FLUX: double/single stream blocks (Kohya "lora_unet_double_blocks_", diffusers "single_transformer_blocks")
    if any(m in joined for m in ("double_blocks", "single_blocks", "single_transformer_blocks", "context_embedder")):
        return "flux"
    sdxl_markers = (
        "lora_te2_", "text_encoder_2", "conditioner.embedders.1",
        "transformer_blocks_9", "transformer_blocks.9",  # SDXL mid/down blocks have 10 blocks
        "add_embedding", "label_emb",
    )
    if any(m in joined for m in sdxl_markers):
        return "sdxl"
    unet_markers = ("lora_unet_", "unet.", "input_blocks", "down_blocks", "mid_block")
    if any(m in joined for m in unet_markers):
        return "sd15"
    if any(k.startswith(("transformer.", "diffusion_model.", "lora_transformer_")) for k in keys):
        return "ernie"
    return "unknown"


EXTRA_DIRS = []  # more folders to look in (Settings -> LoRA folders), set by the backend
FOLDER_FAMILY = {}  # folder -> "sd15" / "sdxl" / "flux" / "ernie": every LoRA in it counts as that type
NAME_FAMILY = {}    # LoRA name -> type you picked for that one LoRA (wins over its folder and the file)
_header_cache = {}  # path -> (mtime, size, family, is_lora, meta)
LORA_KEY_MARKERS = ("lora", "lokr", "hada", "oft_", "dora_scale")


def _file_info(p):
    st = p.stat()
    c = _header_cache.get(str(p))
    if c and c[0] == st.st_mtime and c[1] == st.st_size:
        return c[2], c[3], c[4], st
    try:
        header = read_safetensors_header(p)
        family = detect_family(header)
        meta = header.get("__metadata__") or {}
        is_lora = any(m in k.lower() for k in header if k != "__metadata__" for m in LORA_KEY_MARKERS)
        if not is_lora:
            import embeddings
            if embeddings.classify(header):
                family = "embedding"  # a textual inversion: listed under Embeddings, not as a LoRA
    except Exception:
        family, meta, is_lora = "unknown", {}, False
    _header_cache[str(p)] = (st.st_mtime, st.st_size, family, is_lora, meta)
    return family, is_lora, meta, st


def clip_skip_from_meta(meta):
    """The CLIP skip a LoRA was trained with (Kohya writes ss_clip_skip), or None. Many anime-style
    SD 1.5 LoRAs were trained with 2 and look best with the same setting."""
    try:
        v = int(str((meta or {}).get("ss_clip_skip") or "").strip())
    except ValueError:
        return None
    return v if 1 < v <= 12 else None


def scan(lora_dir, only_loras=False):
    """only_loras: skip files that aren't LoRAs (for shared folders that also hold checkpoints)."""
    out = []
    d = Path(lora_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("**/*.safetensors")):
        if p.name.startswith("._"):
            continue  # macOS resource-fork files on external drives
        try:
            family, is_lora, meta, st = _file_info(p)
        except OSError:
            continue
        if family == "embedding":
            continue  # Settings → Embeddings lists these
        if only_loras and (not is_lora or st.st_size > 2_500_000_000):
            continue
        row = {"name": p.stem, "file": str(p), "family": family, "size_mb": round(st.st_size / 1e6, 1)}
        cs = clip_skip_from_meta(meta)
        if cs:
            row["clip_skip"] = cs
        if family in ("sdxl", "unknown"):
            fl = flavor_from_meta(meta, p.stem)
            if fl:
                row["flavor"] = fl
        if nsfw.name_flag(p.stem) or nsfw.meta_flag(meta):
            row["nsfw_auto"] = True  # content filter: name or training tags say NSFW
        out.append(row)
    return out


def folder_family(path):
    """The type you assigned to the LoRA folder this file is in (the deepest matching folder), or None."""
    best, fam = -1, None
    p = str(path)
    for d, f in FOLDER_FAMILY.items():
        if f in FAMILIES and (p == d or p.startswith(d.rstrip("/") + "/")) and len(d) > best:
            best, fam = len(d), f
    return fam


def all_loras(lora_dir):
    """Your LoRA folders and DiffusionBee's, with each folder's assigned type applied (Settings → LoRA folders)."""
    out = _all_loras(lora_dir)
    if FOLDER_FAMILY or NAME_FAMILY:
        for row in out:
            fam = NAME_FAMILY.get(row["name"]) if NAME_FAMILY.get(row["name"]) in FAMILIES else None
            if fam:
                row["family_set"] = True  # you picked it for this LoRA
            else:
                fam = folder_family(row.get("file", ""))
            if fam and fam != row.get("family"):
                row["detected_family"] = row.get("family")
                row["family"] = fam
    return out


def _all_loras(lora_dir):
    """Your LoRA folder, any extra LoRA folders, plus LoRAs from DiffusionBee (read in place)."""
    out = scan(lora_dir)
    taken = {l["name"] for l in out}
    for extra in EXTRA_DIRS:
        try:
            for l in scan(extra, only_loras=True):
                if l["name"] in taken:
                    continue  # same LoRA already listed
                taken.add(l["name"])
                out.append(dict(l, source="folder"))
        except Exception as e:
            import sys
            print("[loras] couldn't scan", extra, e, file=sys.stderr, flush=True)
    try:
        import diffusionbee
        for i in diffusionbee.loras():
            name = i["name"]
            if name in taken:
                continue  # the original .safetensors is already listed: prefer it
            taken.add(name)
            out.append({"name": name, "file": i["file"], "family": i["family"],
                        "size_mb": i["size_mb"], "source": "diffusionbee",
                        **({"nsfw_auto": True} if nsfw.name_flag(name) else {})})
    except Exception as e:  # never let a bad DiffusionBee file break the LoRA list
        import sys
        print("[loras] DiffusionBee scan failed:", e, file=sys.stderr, flush=True)
    return out


# ------------------------------------------------------------------ trigger words
def _split_words(text):
    if isinstance(text, list):
        items = text
    else:
        items = re.split(r"[,\n]", str(text))
    out = []
    for w in items:
        w = str(w).strip().strip(",").strip()
        if w and len(w) < 120 and w.lower() not in (x.lower() for x in out):
            out.append(w)
    return out


GENERIC_TAGS = {"1girl", "1boy", "solo", "male focus", "female focus", "looking at viewer", "simple background",
                "white background", "smile", "standing", "upper body", "full body", "realistic", "photorealistic",
                "portrait", "outdoors", "indoors", "short hair", "long hair", "black hair", "brown hair", "blonde hair",
                "shirt", "open mouth", "closed mouth", "blush", "teeth", "jewelry", "multiple girls", "multiple boys"}


def triggers_from_meta(meta):
    """(trigger words, suggested tags) from a LoRA's embedded training metadata."""
    words = []
    for key in ("modelspec.trigger_phrase", "ss_trigger_words", "trigger_words", "trigger_phrase", "activation_text"):
        if meta.get(key):
            words += _split_words(meta[key])
    suggested = []
    freq = meta.get("ss_tag_frequency")
    if freq:
        try:
            data = json.loads(freq) if isinstance(freq, str) else freq
            counts = {}
            for tags in data.values():
                for t, n in tags.items():
                    t = t.strip()
                    counts[t] = counts.get(t, 0) + int(n)
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            suggested = [t for t, _ in ranked if t and t.lower() not in GENERIC_TAGS][:8]
        except Exception:
            pass
    return words, suggested


def triggers_from_sidecars(path):
    """Civitai / A1111 helper files saved next to a LoRA: <name>.civitai.info, <name>.json, <name>.txt"""
    p = Path(path)
    words = []
    for side in (p.with_suffix(".civitai.info"), p.with_suffix(".info"), p.with_suffix(".json")):
        try:
            if side.is_file() and side.stat().st_size < 5_000_000:
                data = json.loads(side.read_text(errors="ignore"))
                for key in ("trainedWords", "activation text", "activation_text", "trigger_words", "triggerWords"):
                    if data.get(key):
                        words += _split_words(data[key])
        except Exception:
            pass
    txt = p.with_suffix(".txt")
    try:
        if txt.is_file() and txt.stat().st_size < 2000:
            words += _split_words(txt.read_text(errors="ignore"))
    except Exception:
        pass
    return _split_words(words)


def find_triggers(path, original=None):
    """Look everywhere we can for a LoRA's trigger words. `original` is the source .safetensors a
    DiffusionBee import came from (if it still exists, e.g. on an external drive)."""
    words, suggested = [], []
    for f in [path] + ([original] if original else []):
        if not f:
            continue
        fp = Path(f)
        try:
            if fp.suffix == ".safetensors" and fp.is_file():
                _, _, meta, _ = _file_info(fp)
                w, s = triggers_from_meta(meta)
                words += w
                suggested += s
            words += triggers_from_sidecars(fp)
        except OSError:
            continue
    words = _split_words(words)
    suggested = [s for s in _split_words(suggested) if s.lower() not in (w.lower() for w in words)]
    return {"found": words, "suggested": suggested[:8]}
