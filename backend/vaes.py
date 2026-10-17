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


def list_vaes():
    out = []
    d = paths.VAE_DIR
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in EXTS or p.name.startswith("._"):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < 20e6 or size > 1.2e9:  # a VAE is ~80-340 MB; anything else isn't one
            continue
        out.append({"name": p.stem, "file": str(p), "family": _family(p), "size_mb": round(size / 1e6, 1)})
    return out


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
    vae = AutoencoderKL.from_config(template_vae.config)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"[vae] {os.path.basename(str(path))}: {len(sd)} weights, {len(missing)} missing, {len(unexpected)} unexpected",
          flush=True)
    if len(missing) > 4:
        raise ValueError(f"{os.path.basename(str(path))} doesn't look like a VAE for this model ({len(missing)} parts missing)")
    return vae
