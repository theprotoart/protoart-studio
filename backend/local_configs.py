"""Model "recipes" (config + tokenizer files, a few MB, no weights) so your own checkpoint files load
completely offline.

A single .safetensors checkpoint holds only weights; diffusers also needs the matching model
configuration and tokenizer. Normally it fetches those from Hugging Face every time. Here they are
copied once from a model you already have on this Mac (SD 1.5 / SDXL download or the 8-bit SDXL
copy) into ~/.protoartstudio/configs/<family>, and reused from there with no internet at all.
"""
import json
import shutil
from pathlib import Path

import paths

SOURCES = {
    "sd15": {"repos": ["stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"],
             "local": []},
    "sdxl": {"repos": ["stabilityai/stable-diffusion-xl-base-1.0"], "local": ["sdxl_base_1.0_int8"]},
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".msgpack", ".h5")


def config_dir(family):
    return paths.HOME / "configs" / family


def _copy_recipe(src, dest):
    src, dest = Path(src), Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    n = 0
    for f in src.rglob("*"):
        if not f.is_file() and not f.is_symlink():
            continue
        rel = f.relative_to(src)
        if rel.parts and rel.parts[0] in ("safety_checker",):
            continue
        if f.suffix.lower() in WEIGHT_SUFFIXES:
            continue
        if f.suffix.lower() not in (".json", ".txt", ".model") and not rel.parts[0].startswith("tokenizer"):
            continue
        out = tmp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)  # follows the HF cache's symlinks
        if out.suffix == ".json":
            try:  # an 8-bit copy's configs mention its quantization: a plain checkpoint isn't quantized
                data = json.loads(out.read_text())
                if isinstance(data, dict) and "quantization_config" in data:
                    data.pop("quantization_config")
                    out.write_text(json.dumps(data, indent=2))
            except Exception:
                pass
        n += 1
    if not (tmp / "model_index.json").exists():
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return n > 0


INPAINT_PIPES = {"sd15": "StableDiffusionInpaintPipeline", "sdxl": "StableDiffusionXLInpaintPipeline"}


def ensure_inpaint(family):
    """The same recipe for an inpainting checkpoint: its UNet takes 9 input channels (picture latents + mask +
    masked picture) instead of 4. Made from the normal recipe, on this Mac, once."""
    base = ensure(family)
    if base is None:
        return None
    dest = config_dir(family + "_inpaint")
    if (dest / "model_index.json").exists():
        return str(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base, tmp)
    unet_cfg = tmp / "unet" / "config.json"
    data = json.loads(unet_cfg.read_text())
    data["in_channels"] = 9
    unet_cfg.write_text(json.dumps(data, indent=2))
    idx = tmp / "model_index.json"
    data = json.loads(idx.read_text())
    data["_class_name"] = INPAINT_PIPES[family]
    idx.write_text(json.dumps(data, indent=2))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return str(dest)


def ensure(family):
    """Path of a local recipe folder for sd15/sdxl, creating it from local files if needed.
    Returns None for other families. Raises an 'offline' error if it must be downloaded first."""
    if family not in SOURCES:
        return None
    dest = config_dir(family)
    if (dest / "model_index.json").exists():
        return str(dest)
    src = SOURCES[family]
    for name in src["local"]:
        d = paths.QUANT_DIR / name
        if (d / "model_index.json").exists() and _copy_recipe(d, dest):
            return str(dest)
    from huggingface_hub import snapshot_download
    patterns = ["*.json", "*.txt", "tokenizer*/*"]
    for repo in src["repos"]:
        try:
            folder = snapshot_download(repo, local_files_only=True, allow_patterns=patterns)
            if _copy_recipe(folder, dest):
                return str(dest)
        except Exception:
            continue
    # Nothing on this Mac yet: fetch just the small recipe files (allowed only when online).
    import net
    if not net.is_online():
        raise RuntimeError(f"offline: the {family} model recipe (a few MB of config files) isn't on this Mac yet")
    folder = snapshot_download(src["repos"][0], allow_patterns=patterns)
    if _copy_recipe(folder, dest):
        return str(dest)
    raise RuntimeError(f"Couldn't prepare the {family} model recipe.")
