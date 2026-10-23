"""VAEs: the part of an SD 1.5 / SDXL model that turns its internal picture into pixels. A different VAE changes colour
and fine detail (e.g. "vae-ft-mse-840000" fixes washed-out or blotchy SD 1.5 pictures; "sdxl-vae-fp16-fix" avoids
black pictures in half precision). Files live in ~/.protoartstudio/vaes (Civitai downloads go there too) and are
read where they are. Only .safetensors and .pt (read with torch's weights-only loader, which can't run code).
"""
import json
import os
from pathlib import Path

import paths

EXTS = (".safetensors", ".pt")


def _family(p: Path):
    """SD 1.5 or SDXL: from Civitai's details beside the file, else from the name."""
    info = p.with_name(p.stem + ".civitai.info")
    try:
        base = json.loads(info.read_text()).get("baseModel") or ""
        import civitai
        fam = civitai.family_of(base)[0]
        if fam in ("sd15", "sdxl"):
            return fam
    except (OSError, ValueError, ImportError, AttributeError):
        pass
    low = p.stem.lower()
    return "sdxl" if ("xl" in low or "pony" in low or "illustrious" in low or "noob" in low) else "sd15"


EXTRA_DIRS = []  # VAE folders of a model library you use (Stability Matrix, ComfyUI…), read in place


def list_vaes():
    out, taken = [], set()
    from pathlib import Path
    files = []
    for d in [paths.VAE_DIR] + [Path(x) for x in EXTRA_DIRS]:
        if d.is_dir():
            try:
                files += sorted(d.rglob("*"))
            except OSError:
                pass
    for p in files:
        if p.stem in taken:
            continue
        if p.suffix.lower() not in EXTS or p.name.startswith("._"):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < 20e6 or size > 1.2e9:  # a VAE is ~80-340 MB; anything else isn't one
            continue
        taken.add(p.stem)
        out.append({"name": p.stem, "file": str(p), "family": _family(p), "size_mb": round(size / 1e6, 1)})
    return out


# The usual VAE for each model type (both MIT licence), offered as one click in Create → VAE.
RECOMMENDED = {
    "sd15": {"repo": "stabilityai/sd-vae-ft-mse-original", "revision": "19abcd2d75bb1b188ea5fad5c80799b59c1ef974",
             "files": ["vae-ft-mse-840000-ema-pruned.safetensors"], "name": "vae-ft-mse-840000-ema-pruned", "mb": 335,
             "why": "sharper faces and fewer washed-out colours than SD 1.5's own"},
    "sdxl": {"repo": "madebyollin/sdxl-vae-fp16-fix", "revision": "6d1073461cd0b5a6ea4fda10b812e3d9d58a8330",
             "files": ["sdxl_vae.safetensors", "sdxl.vae.safetensors"], "name": "sdxl-vae-fp16-fix", "mb": 335,
             "why": "the SDXL VAE fixed to work in half precision (no black or speckled pictures)"},
}


def get_recommended(family, emit=lambda m: None):
    """Download the recommended VAE for SD 1.5 or SDXL into the VAE folder (once). Returns its row."""
    import shutil
    from huggingface_hub import hf_hub_download
    rec = RECOMMENDED.get(family)
    if rec is None:
        raise ValueError("there's no recommended VAE for this model type")
    have = find(rec["name"])
    if have:
        return have
    emit({"type": "status", "msg": f"Downloading {rec['name']} (~{rec['mb']} MB, first time only)…"})
    last = None
    for fname in rec["files"]:
        try:
            src = hf_hub_download(rec["repo"], fname, revision=rec["revision"])
            break
        except Exception as e:  # the file may have another name in this repository: try the next
            last = e
    else:
        raise last
    paths.VAE_DIR.mkdir(parents=True, exist_ok=True)
    dest = paths.VAE_DIR / (rec["name"] + ".safetensors")
    shutil.copyfile(src, dest)
    return find(rec["name"])


OWN = "__own__"  # Create → VAE "the model's own" (an empty choice means Automatic)


def is_fp16_fix(name):
    low = (name or "").lower().replace("_", "-")
    return "fp16-fix" in low or "fp16fix" in low


def has_own_vae(model_file):
    """Does this checkpoint file carry a VAE? (Read from the file's index only: nothing is loaded.)
    True when it can't tell (not a .safetensors file, unreadable), so the model's own is used."""
    try:
        import struct
        with open(model_file, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > 100_000_000:
                return True
            keys = json.loads(fh.read(n)).keys()
        return any(k.startswith(("first_stage_model.", "vae.")) for k in keys)
    except Exception:
        return True


def auto_for(model_file, family, rows=None):
    """The VAE Create's "Automatic" picks for a model: (name or None for the model's own, reason).
    1. the VAE its Civitai page comes with (downloaded with it), 2. SDXL: the fp16-fix VAE (the standard SDXL VAE
    breaks in half precision, so the model's own is decoded slowly at full precision), 3. an SD 1.5 file with no
    VAE inside: vae-ft-mse. Otherwise the model's own. Only VAEs already in the VAE folder are used."""
    if family not in ("sd15", "sdxl"):
        return None, ""
    rows = [v for v in (rows if rows is not None else list_vaes()) if v["family"] == family]
    by_name = {v["name"]: v for v in rows}
    if model_file:
        try:
            info = json.loads(Path(model_file).with_name(Path(model_file).stem + ".civitai.info").read_text())
            if info.get("vae") in by_name:
                return info["vae"], "the VAE this model comes with"
        except (OSError, ValueError):
            pass
    if family == "sdxl":
        fix = next((v["name"] for v in rows if is_fp16_fix(v["name"])), None)
        if fix:
            return fix, "SDXL's VAE fixed for half precision: no black pictures, faster and lighter to decode"
        return None, "the model's own (get sdxl-vae-fp16-fix for faster, safer decoding)"
    if model_file and str(model_file).lower().endswith(".safetensors") and not has_own_vae(model_file):
        mse = next((v["name"] for v in rows if "ft-mse" in v["name"].lower().replace("_", "-")
                    or "ftmse" in v["name"].lower()), None)
        if mse:
            return mse, "this model file has no VAE inside"
        return None, "this model file has no VAE inside: get vae-ft-mse-840000 (Create → VAE)"
    return None, "the model's own"


def find(name):
    return next((v for v in list_vaes() if v["name"] == name), None)


def read_state(path):
    """The file's weights, in diffusers' layout."""
    import torch
    if str(path).lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
    return sd


def build(path, template_vae):
    """An AutoencoderKL like the model's own, with this file's weights. template_vae: the pipeline's VAE (gives the
    configuration, so nothing has to be downloaded)."""
    from diffusers import AutoencoderKL
    sd = read_state(path)
    if not any(k.startswith(("encoder.down_blocks", "decoder.up_blocks")) for k in sd):  # original (LDM) layout
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        sd = convert_ldm_vae_checkpoint(sd, dict(template_vae.config))
    cfg = dict(template_vae.config)
    if is_fp16_fix(Path(str(path)).stem):
        cfg["force_upcast"] = False  # made for half precision: decode in fp16 (no slow, memory-hungry fp32 step)
    vae = AutoencoderKL.from_config(cfg)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"[vae] {os.path.basename(str(path))}: {len(sd)} weights, {len(missing)} missing, {len(unexpected)} unexpected",
          flush=True)
    if len(missing) > 4:
        raise ValueError(f"{os.path.basename(str(path))} doesn't look like a VAE for this model ({len(missing)} parts missing)")
    return vae
