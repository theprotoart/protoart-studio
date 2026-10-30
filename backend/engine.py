"""Image generation engine built on Hugging Face diffusers.

One pipeline is kept loaded at a time. Switching models unloads the previous one
to free unified memory. LoRAs are loaded as named adapters so a stack of them can
be mixed with individual weights and swapped without reloading the base model.
"""
import gc
import math
import json
import os
import random
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import nsfw as nsfw_words
import paths
import kv_cache
import samplers
import thumbs
import loras as lora_scan
import model_hub

# ---------------------------------------------------------------- model catalogue
MODELS = {
    "sd15": {
        "label": "Stable Diffusion 1.5", "family": "sd15",
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "variant": "fp16",
        "download_gb": 4.3, "min_ram_gb": 8,
        "defaults": {"width": 512, "height": 512, "steps": 25, "cfg": 7.0},
    },
    "sdxl_int8": {
        "label": "SDXL Base 1.0 · 8-bit", "family": "sdxl",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0", "variant": "fp16",
        # First load downloads the fp16 release (~7 GB), converts the UNet to 8-bit (SDNQ)
        # and saves that copy (~4.4 GB) under ~/.beestudio/quantized; later loads use the copy.
        "quantize_unet": "int8", "local_name": "sdxl_base_1.0_int8",
        "download_gb": 7.0, "disk_gb": 4.4, "min_ram_gb": 8,
        "defaults": {"width": 1024, "height": 1024, "steps": 30, "cfg": 6.0},
    },
    # The original full-precision SDXL (the same download the 8-bit build is made from). Needed to train
    # SDXL LoRAs; also usable for pictures (Low-memory mode is switched on automatically under 16 GB).
    "sdxl_base_fp16": {
        "label": "SDXL Base 1.0 · full precision", "family": "sdxl",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0", "variant": "fp16",
        "download_gb": 7.0, "min_ram_gb": 16, "low_memory_min_ram_gb": 8,
        "defaults": {"width": 1024, "height": 1024, "steps": 30, "cfg": 6.0},
    },
    # ERNIE-Image builds (all quantized). Sizes exclude the optional prompt enhancer ("pe"),
    # which is only downloaded the first time you switch it on.
    "ernie_turbo_q6": {
        "label": "ERNIE-Image Turbo · 6-bit", "family": "ernie", "quant": "Q6_K",
        # transformer: Unsloth GGUF Q6_K; text encoder / VAE / tokenizer: 4-bit SDNQ build
        "repo": "OzzyGT/ERNIE_Image_Turbo_sdnq_dynamic_int4",
        "gguf_repo": "unsloth/ERNIE-Image-Turbo-GGUF", "gguf_file": "ernie-image-turbo-Q6_K.gguf",
        "config_repo": "baidu/ERNIE-Image-Turbo",
        "download_gb": 9.8, "pe_gb": 3.3, "min_ram_gb": 16,
        "defaults": {"width": 1024, "height": 1024, "steps": 8, "cfg": 1.0},
    },
    # Same model on the fast engine (MLX, like Draw Things): native 4-bit Apple GPU kernels.
    "ernie_turbo_mlx_q4": {
        "label": "ERNIE-Image Turbo · 4-bit · ⚡ fast engine", "family": "ernie", "quant": "Q4 MLX",
        "engine": "mlx", "mlx_repo": "mflux-community/ernie-image-turbo-mflux-q4", "mlx_base": "ernie-image-turbo",
        "download_gb": 6.6, "engine_gb": 1.5, "min_ram_gb": 8,
        "defaults": {"width": 1024, "height": 1024, "steps": 8, "cfg": 1.0},
    },
    # MOSS-Image (OpenMOSS, Apache-2.0): a 5B model made for lettering and layouts (signs, posters, covers),
    # on the Apple GPU with moss-image-mlx in its own Python. See moss_engine.py.
    "moss_turbo_mlx": {
        "label": "MOSS-Image · Turbo · ⚡ lettering & posters", "family": "moss", "quant": "8-bit MLX",
        "engine": "mlx", "moss": True, "moss_turbo": True,
        "mlx_repo": "VisualInference/MOSS-Image-SFT-MLX-compact", "mlx_revision": "310ab44798e3711eb134c1b7d30a73083c91af58",
        "mlx_patterns": ["*.json", "*.jinja", "LICENSE", "NOTICE", "scheduler/*", "text_encoder/*", "tokenizer/*",
                         "transformer/*", "vae/*"],
        "download_gb": 12.7, "engine_gb": 1.0, "min_ram_gb": 16, "license": "Apache-2.0",
        "defaults": {"width": 1024, "height": 1024, "steps": 15, "cfg": 3.0},
    },
    "moss_mlx": {
        "label": "MOSS-Image · best quality (50 steps, slow)", "family": "moss", "quant": "8-bit MLX",
        "engine": "mlx", "moss": True,
        "mlx_repo": "VisualInference/MOSS-Image-SFT-MLX-compact", "mlx_revision": "310ab44798e3711eb134c1b7d30a73083c91af58",
        "mlx_patterns": ["*.json", "*.jinja", "LICENSE", "NOTICE", "scheduler/*", "text_encoder/*", "tokenizer/*",
                         "transformer/*", "vae/*"],
        "download_gb": 12.1, "engine_gb": 1.0, "min_ram_gb": 16, "license": "Apache-2.0",
        "defaults": {"width": 1024, "height": 1024, "steps": 50, "cfg": 3.0},
    },
    "ernie_turbo_int4": {
        "label": "ERNIE-Image Turbo · 4-bit (smallest)", "family": "ernie", "quant": "int4",
        "repo": "OzzyGT/ERNIE_Image_Turbo_sdnq_dynamic_int4",
        "download_gb": 7.6, "pe_gb": 3.3, "min_ram_gb": 16,
        "defaults": {"width": 1024, "height": 1024, "steps": 8, "cfg": 1.0},
    },
    "ernie_turbo_int8": {
        "label": "ERNIE-Image Turbo · 8-bit", "family": "ernie", "quant": "int8",
        "repo": "OzzyGT/ERNIE_Image_Turbo_sdnq_dynamic_int8",
        "download_gb": 12.6, "pe_gb": 4.6, "min_ram_gb": 24,
        "defaults": {"width": 1024, "height": 1024, "steps": 8, "cfg": 1.0},
    },
    # FLUX.1 [schnell] by Black Forest Labs (Apache-2.0), 4 steps, on the fast engine (MLX).
    # Community MLX conversions made with the same mflux version the fast engine uses.
    "flux_schnell_mlx_q3": {
        "label": "FLUX.1 Schnell · 3-bit · ⚡ fast engine (for 8 GB Macs)", "family": "flux", "quant": "Q3 MLX",
        "engine": "mlx", "mlx_repo": "mflux-community/flux-1-schnell-mflux-q3", "mlx_base": "schnell",
        "download_gb": 7.5, "engine_gb": 1.5, "min_ram_gb": 8, "license": "Apache-2.0",
        "defaults": {"width": 1024, "height": 1024, "steps": 4, "cfg": 0.0},
    },
    "flux_schnell_mlx_q4": {
        "label": "FLUX.1 Schnell · 4-bit · ⚡ fast engine", "family": "flux", "quant": "Q4 MLX",
        "engine": "mlx", "mlx_repo": "mflux-community/flux-1-schnell-mflux-q4", "mlx_base": "schnell",
        "download_gb": 9.6, "engine_gb": 1.5, "min_ram_gb": 16, "license": "Apache-2.0",
        "defaults": {"width": 1024, "height": 1024, "steps": 4, "cfg": 0.0},
    },
    "flux_schnell_mlx_q8": {
        "label": "FLUX.1 Schnell · 8-bit · ⚡ fast engine (best quality)", "family": "flux", "quant": "Q8 MLX",
        "engine": "mlx", "mlx_repo": "mflux-community/flux-1-schnell-mflux-q8", "mlx_base": "schnell",
        "download_gb": 18.0, "engine_gb": 1.5, "min_ram_gb": 32, "license": "Apache-2.0",
        "defaults": {"width": 1024, "height": 1024, "steps": 4, "cfg": 0.0},
    },
    # FLUX.1 [dev] by Black Forest Labs: higher quality than Schnell (25 steps), for Macs with lots of memory. The
    # official files (gated: accept the licence on Hugging Face and save a token in Settings → Models) are shrunk to
    # 8-bit by the fast engine as they load. FLUX.1 [dev] Non-Commercial License: pictures for personal use only.
    "flux_dev_mlx_q8": {
        "label": "FLUX.1 Dev · 8-bit · ⚡ fast engine (32 GB Macs, non-commercial licence)", "family": "flux",
        "quant": "Q8 MLX", "engine": "mlx", "mlx_repo": "black-forest-labs/FLUX.1-dev", "mlx_base": "dev",
        "mlx_quantize": 8, "mlx_patterns": ["model_index.json", "transformer/*", "text_encoder/*", "text_encoder_2/*",
                                             "tokenizer/*", "tokenizer_2/*", "vae/*", "scheduler/*"],
        "download_gb": 33.5, "engine_gb": 1.5, "min_ram_gb": 32, "license": "FLUX.1 [dev] Non-Commercial",
        "gated": True,
        "defaults": {"width": 1024, "height": 1024, "steps": 25, "cfg": 3.5},
    },
    # FLUX.2 [klein] 4B by Black Forest Labs (Apache-2.0, January 2026): a newer, smaller FLUX that follows
    # prompts and writes text better than FLUX.1 Schnell, in 4 steps. Its text encoder is Qwen3 (sentences in
    # many languages). A community 4-bit MLX conversion (~4.6 GB), pinned to one revision so the files can't
    # change under us; the fast engine drops the text encoder while it draws, so it fits an 8 GB Mac.
    "flux2_klein_mlx_q4": {
        "label": "FLUX.2 Klein 4B · 4-bit · ⚡ fast engine (newest, for 8 GB Macs)", "family": "flux2",
        "quant": "Q4 MLX", "engine": "mlx", "mlx_repo": "ar9av/FLUX.2-klein-4B-mflux-4bit",
        "mlx_revision": "86214a1c1caaea43b7c25b452b74ec08cd33d3e7", "mlx_base": "flux2-klein-4b",
        "download_gb": 4.6, "engine_gb": 1.5, "min_ram_gb": 8, "license": "Apache-2.0",
        "defaults": {"width": 1024, "height": 1024, "steps": 4, "cfg": 1.0},
    },
    # Anima (CircleStone Labs, 2B, anime / illustration; tags or sentences). Official diffusers files; the Turbo
    # build uses the official distilled transformer. CircleStone Labs Non-Commercial License (pictures may be sold).
    "anima_turbo": {
        "label": "Anima Turbo · anime (non-commercial licence)", "family": "anima", "anima_turbo": True,
        "download_gb": 5.9, "min_ram_gb": 8, "license": "CircleStone Labs Non-Commercial",
        "defaults": {"width": 768, "height": 1024, "steps": 10, "cfg": 1.0},
    },
    "anima_base": {
        "label": "Anima · anime, 30 steps (non-commercial licence)", "family": "anima",
        "download_gb": 5.6, "min_ram_gb": 12, "license": "CircleStone Labs Non-Commercial",
        "defaults": {"width": 768, "height": 1024, "steps": 30, "cfg": 4.5},
    },
    "ernie_int4": {
        "label": "ERNIE-Image · 4-bit (50 steps, best quality)", "family": "ernie", "quant": "int4",
        "repo": "OzzyGT/ERNIE_Image_sdnq_dynamic_int4",
        "download_gb": 7.6, "pe_gb": 3.3, "min_ram_gb": 16,
        "defaults": {"width": 1024, "height": 1024, "steps": 50, "cfg": 4.0},
    },
}

# "Fast mode": official LCM speed LoRAs (4-8 steps instead of 25-30)
FAST_LORAS = {
    "sd15": ("latent-consistency/lcm-lora-sdv1-5", "pytorch_lora_weights.safetensors", 0.14),
    "sdxl": ("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", 0.4),
}
FAST_ADAPTER = "fast_lcm"

ERNIE_SIZES = [(1024, 1024), (848, 1264), (1264, 848), (768, 1376), (896, 1200), (1376, 768), (1200, 896)]
PE_EXTRA_RAM_GB = 6  # the ERNIE prompt enhancer is an extra language model
LOW_MEMORY_SAVES_GB = {"sd15": 2, "sdxl": 6}  # rough RAM saved by int8 low-memory mode


class Cancelled(Exception):
    pass


class Skipped(Exception):
    """The current image was skipped; the job carries on with its next image."""


class UserError(Exception):
    """An error with a message meant for the person using the app."""


def total_ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _moss_sizes():
    import moss_engine
    return moss_engine.SIZES


def _readable(name):
    """A file name as people write it: DiffusionBee keeps web-style escapes from downloads ("Kim%20Jung" -> "Kim Jung")."""
    from urllib.parse import unquote
    try:
        return re.sub(r"\s+", " ", unquote(str(name))).strip() or str(name)
    except Exception:
        return str(name)


def free_disk_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return round(shutil.disk_usage(p).free / 2**30, 1)


def _slug(text, n=40):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")
    return (s[:n] or "image")


def _adapter_id(name):
    return "l_" + re.sub(r"\W", "_", name)


def _custom_models():
    """User checkpoints dropped into ~/.beestudio/models/<file>.safetensors."""
    out = {}
    for p in sorted(paths.MODELS_DIR.glob("*.safetensors")):
        try:
            header = lora_scan.read_safetensors_header(p)
            keys = "\n".join(k for k in header if k != "__metadata__")
            family = "sdxl" if "conditioner.embedders.1" in keys else "sd15"
            meta = header.get("__metadata__") or {}
        except Exception:
            continue
        base = MODELS["sdxl_int8" if family == "sdxl" else "sd15"]
        inpaint = model_hub.is_inpaint_header(header)
        out["file:" + p.name] = {
            "label": f"{p.stem} ({'SDXL' if family == 'sdxl' else 'SD 1.5'} {'inpainting' if inpaint else 'checkpoint'})",
            **({"inpaint": True} if inpaint else {}),
            "family": family, "file": str(p), "download_gb": 0, "quant": None,
            # full-precision checkpoints: SDXL needs Low-memory mode under 16 GB
            "min_ram_gb": 16 if family == "sdxl" else 8, "low_memory_min_ram_gb": 8 if family == "sdxl" else None,
            "defaults": dict(base["defaults"]),
            **({"flavor": lora_scan.flavor_from_meta(meta, p.stem)} if family == "sdxl" else {}),
        }
    # DiffusionBee models (.tdict), read in place
    try:
        import diffusionbee
        for i in diffusionbee.checkpoints():
            inpaint = "inpaint" in i["name"].lower()
            out["tdict:" + Path(i["file"]).name] = {
                "label": f"{_readable(i['name'])} (DiffusionBee{' inpainting' if inpaint else ''})", "family": "sd15", "tdict": i["file"],
                **({"inpaint": True} if inpaint else {}),
                "download_gb": 0, "quant": None, "min_ram_gb": 8, "source": "diffusionbee",
                "defaults": dict(MODELS["sd15"]["defaults"]),
            }
    except Exception as e:
        print("[engine] DiffusionBee scan failed:", e, flush=True)
    return out


# Names of anime-style SD 1.5 models (most descend from the NovelAI model and were made with CLIP skip 2)
_ANIME_HINTS = ("anime", "anything", "counterfeit", "meina", "meinamix", "abyss", "orangemix", "aom", "waifu",
                "novelai", "nai", "pastel", "cetus", "hassaku", "darksushi", "dark sushi", "ghostmix", "mistoon",
                "toon", "manga", "kawaii", "revanimated", "rev animated", "aniverse", "2.5d", "2d", "hentai",
                "niji", "nijijourney", "kotos", "breakdomain", "grapefruit", "andite", "7th", "bluepencil",
                "cuteyukimix", "yuki", "kenshi")


# Quality tags each SDXL flavour was trained with (the community-standard ones).
QUALITY_TAGS = {
    "pony": {"prompt": "score_9, score_8_up, score_7_up", "negative": "score_6, score_5, score_4"},
    "illustrious": {"prompt": "masterpiece, best quality, amazing quality, very aesthetic, absurdres",
                    "negative": "worst quality, low quality, bad quality, lowres"},
    "noobai": {"prompt": "masterpiece, best quality, newest, absurdres, highres",
               "negative": "worst quality, old, early, low quality, lowres"},
}


def model_flavor(spec, model_id=""):
    """Pony / Illustrious / NoobAI for SDXL models (from the name, or the checkpoint's own metadata)."""
    if spec.get("family") != "sdxl" or spec.get("engine"):
        return None
    if spec.get("flavor"):
        return spec["flavor"]
    return lora_scan.detect_flavor(model_id, spec.get("label"), spec.get("file"), spec.get("local_dir"),
                                   spec.get("repo"))


def add_quality_tags(text, tags):
    """Put the quality tags in front of a prompt, unless it already starts with (some of) them."""
    text = (text or "").strip()
    first = tags.split(",")[0].strip().lower()
    if not tags or first in text.lower():
        return text
    return f"{tags}, {text}" if text else tags


def recommended_clip_skip(spec, model_id=""):
    """CLIP skip this model is meant to be used with, in the numbering other apps use (1 = off).
    SDXL: 2 is the normal way SDXL reads prompts (what diffusers does by default).
    SD 1.5: 2 for anime-style models (NovelAI lineage), 1 for photo/general models."""
    fam = spec.get("family")
    if fam == "sdxl":
        return 2
    if fam != "sd15" or spec.get("engine"):
        return None
    meta_cs = spec.get("meta_clip_skip")
    if meta_cs:
        return meta_cs
    text = " ".join(str(x) for x in (model_id, spec.get("label"), spec.get("file"), spec.get("tdict"),
                                     spec.get("local_dir"), spec.get("repo")) if x).lower()
    text = re.sub(r"[_\-.]+", " ", text)
    words = set(text.split())
    for h in _ANIME_HINTS:
        if (" " in h and h in text) or h in words or (len(h) > 4 and h in text.replace(" ", "")):
            return 2
    return 1


_PATH_KEYS = ("init_image", "control_image", "edited_from", "image", "path", "file")


def _shareable(value, key=""):
    """Copy of image settings without local file paths: absolute paths become file names, and
    model ids such as 'local:/Users/…/x.safetensors' keep only the file name."""
    if isinstance(value, dict):
        return {k: _shareable(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_shareable(v, key) for v in value]
    if isinstance(value, str):
        for prefix in ("local:", "file:", "tdict:", "hf:"):
            if value.startswith(prefix) and ("/" in value[len(prefix):] and value[len(prefix):].startswith(("/", "~"))):
                return prefix + os.path.basename(value[len(prefix):].rstrip("/"))
        if value.startswith(("/", "~/")) or (key in _PATH_KEYS and os.sep in value):
            return os.path.basename(value.rstrip("/"))
    return value


def _hires_cap(family):
    """Largest side Hi-res fix goes to on this Mac."""
    ram = total_ram_gb() or 8
    if family == "sd15":
        return 1024 if ram < 24 else 1536
    return 1536 if ram < 24 else 2048


def _hf_file_cached(repo, filename):
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, filename, local_files_only=True)
        return True
    except Exception:
        return False


def _max_side(family):
    """Largest side used when an image-to-image result takes the picture's own size. A big photo at full
    size (e.g. 1536 px with SD 1.5) needs far more memory than a small Mac has (one run asked for 20 GB)."""
    ram = total_ram_gb() or 8
    if family == "sd15":
        return 768 if ram < 16 else 1024
    return 1024 if ram < 24 else 1536


def _slice_self_attention(unet, slice_size=2):
    """Attention slicing for the self-attention layers only (the image prompt adapter's cross-attention layers stay
    as they are). Same pictures, much less memory for the big attention tables."""
    if unet is None:
        return
    from diffusers.models.attention_processor import SlicedAttnProcessor
    procs = unet.attn_processors
    if not any(".attn1." in k and type(p).__name__ in ("AttnProcessor2_0", "AttnProcessor") for k, p in procs.items()):
        return
    unet.set_attn_processor({k: (SlicedAttnProcessor(slice_size) if ".attn1." in k and
                                 type(p).__name__ in ("AttnProcessor2_0", "AttnProcessor") else p)
                             for k, p in procs.items()})


def _set_seamless(pipe, on):
    """Make every convolution in the UNet and VAE wrap around the edges (A1111's "Tiling"), or back to normal."""
    import torch
    mode = "circular" if on else "zeros"
    for part in (getattr(pipe, "unet", None), getattr(pipe, "vae", None)):
        if part is None:
            continue
        for m in part.modules():
            if isinstance(m, torch.nn.Conv2d) and m.padding_mode != mode:
                m.padding_mode = mode
                m._reversed_padding_repeated_twice = torch.nn.modules.utils._reverse_repeat_tuple(m.padding, 2)


def _slerp(t, a, b):
    """Spherical blend of two noise tensors (keeps the noise's strength, unlike a straight mix)."""
    import torch
    af, bf = a.flatten().double(), b.flatten().double()
    dot = torch.clamp((af / af.norm()) @ (bf / bf.norm()), -1.0, 1.0)
    if abs(float(dot)) > 0.9995:
        return ((1 - t) * a.double() + t * b.double()).to(a.dtype)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) / so) * a.double() + (torch.sin(t * omega) / so) * b.double()
    return out.to(a.dtype)


def variation_noise(seed, var_seed, strength, shape, dtype):
    """Starting noise for "Small change" (a variation seed): the seed's own noise, nudged towards another seed's.
    Strength 0 is exactly the normal picture; 0.05–0.2 changes small details; 1 is the other seed's picture."""
    import torch
    # made exactly the way diffusers makes it (CPU generator, the model's dtype), so strength 0 is the same picture
    base = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(int(seed)), dtype=dtype)
    if strength <= 0:
        return base
    other = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(int(var_seed)), dtype=dtype)
    return _slerp(float(strength), base, other).to(dtype)


def _looks_blank(image):
    """True for a (nearly) single-colour image."""
    try:
        import numpy as np
        a = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
        return float(a.std()) < 1.5
    except Exception:
        return False


class _TextModelAlias:
    """Stands in for `text_encoder.text_model`. Newer transformers (5.x) flattened CLIPTextModel, so the
    final layer norm sits on the model itself, but diffusers' CLIP-skip code still looks for
    `text_encoder.text_model.final_layer_norm`. Not an nn.Module, so it isn't saved or moved twice."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)


def _clip_skip_compat(pipe):
    for name in ("text_encoder", "text_encoder_2"):
        te = getattr(pipe, name, None)
        if te is None or "text_model" in te.__dict__ or hasattr(type(te), "text_model"):
            continue
        if "text_model" in getattr(te, "_modules", {}):
            continue  # older transformers: the real submodule is there
        if hasattr(te, "final_layer_norm"):
            object.__setattr__(te, "text_model", _TextModelAlias(te))


def _repair_meta_tensors(pipe):
    """Single-file loading builds layers as empty 'meta' tensors and fills them from the file. Anything the
    file doesn't contain stays empty and makes .to(device) fail with "Cannot copy out of meta tensor".
    Rebuild known computed buffers (e.g. CLIP's position ids); report anything else clearly."""
    import torch
    broken = []
    for cname, comp in getattr(pipe, "components", {}).items():
        if not isinstance(comp, torch.nn.Module):
            continue
        for mname, mod in comp.named_modules():
            for bname, buf in list(mod.named_buffers(recurse=False)):
                if buf is not None and buf.is_meta:
                    if bname == "position_ids":
                        n = buf.shape[-1]
                        mod.register_buffer(bname, torch.arange(n).expand(1, -1), persistent=False)
                    else:
                        broken.append(f"{cname}.{mname}.{bname}")
            for pname, par in list(mod.named_parameters(recurse=False)):
                if par.is_meta:
                    broken.append(f"{cname}.{mname}.{pname}")
    if broken:
        print("[engine] weights missing from the model file:", broken[:40], flush=True)
        raise UserError(f"This model file is missing {len(broken)} weight(s) (e.g. {broken[0]}), so it can't be "
                        "loaded. It may be incomplete or a different model type.")


def _hub_dir():
    """The downloaded-models folder (or the nearest folder that exists, for measuring free space)."""
    p = paths.hub_cache()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


class _quantized_unet_ok:
    """While the image prompt adapter is added to an 8-bit (quantized) UNet: diffusers then calls
    unet.to(dtype=..., device=...), which quantized models refuse ("Casting a quantized model to a new dtype
    is unsupported"). Inside this block that call only moves the model to its device, and afterwards the
    adapter's own (unquantized) layers are given the model's working precision."""

    def __init__(self, unet):
        self.unet = unet
        self.quantized = bool(getattr(unet, "is_quantized", False) or getattr(unet, "hf_quantizer", None)
                              or getattr(getattr(unet, "config", None), "quantization_config", None))

    def __enter__(self):
        if self.quantized:
            import torch
            unet = self.unet

            def to(*a, **kw):
                kw.pop("dtype", None)
                a = tuple(x for x in a if not isinstance(x, torch.dtype))
                return torch.nn.Module.to(unet, *a, **kw)
            self.dtype = next((p.dtype for p in unet.parameters() if p.is_floating_point()), torch.float16)
            unet.to = to  # this object only, removed again on exit
        return self

    def __exit__(self, *exc):
        if self.quantized:
            unet = self.unet
            try:
                del unet.to
            except AttributeError:
                pass
            if exc[0] is None:
                dev = next(unet.parameters()).device
                for mod in [getattr(unet, "encoder_hid_proj", None)] + list(unet.attn_processors.values()):
                    if hasattr(mod, "to") and hasattr(mod, "parameters"):
                        mod.to(device=dev, dtype=self.dtype)
        return False


def _free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0


def _swap_used_gb():
    try:
        import psutil
        return psutil.swap_memory().used / 1e9
    except Exception:
        return -1.0


def _pipe_dtype(pipe):
    for name in ("unet", "transformer"):
        m = getattr(pipe, name, None)
        if m is not None:
            try:
                return next(p.dtype for p in m.parameters() if p.is_floating_point())
            except StopIteration:
                pass
    return None


def _keep_dtype(pipe):
    """from_pipe() in recent diffusers converts the whole pipeline to float32 unless told otherwise, and the parts are
    shared with the loaded model: one ControlNet / inpaint / image-to-image job silently doubled the model's memory
    (fp16 -> fp32), which on 8 GB Macs meant swapping and 20-60 s a step. Keep the precision the model has."""
    dt = _pipe_dtype(pipe)
    # torch_dtype works in older and newer diffusers alike (newer ones also accept dtype)
    return {"torch_dtype": dt} if dt is not None else {}


def _from_tdict(cls, path, **kw):
    """Build a pipeline (or model) from a DiffusionBee .tdict via diffusers' single-file loader.
    The loader normally reads .safetensors/.ckpt; we hand it the tdict's weights instead."""
    import tdict_reader
    import diffusers.loaders.single_file as sf
    # Read the weights straight into the pipeline's precision (fp16 on Apple GPUs): a model stored in fp32
    # would otherwise sit in memory twice at full size while loading, which on an 8 GB Mac pushes it into
    # swap and makes every step many times slower.
    dtype = kw.get("torch_dtype")
    td = tdict_reader.TDict(path)
    state = {}
    for k in td.checkpoint_keys():
        t = td.tensor(k)
        state[k] = t.to(dtype) if dtype is not None and t.is_floating_point() else t
    original = sf.load_single_file_checkpoint
    sf.load_single_file_checkpoint = lambda *a, **k: state
    try:
        return cls.from_single_file(path, **kw)
    finally:
        sf.load_single_file_checkpoint = original
        state.clear()  # hand the CPU copy back right away
        del state, td
        gc.collect()


def all_models():
    m = dict(MODELS)
    m.update(_custom_models())
    for lm in list(model_hub.LOCAL_MODELS) + model_hub.scan_model_dirs():  # your own files/folders (offline)
        if lm.get("path") and not os.path.exists(lm["path"]):
            continue  # moved to the Trash, renamed or on a drive that isn't plugged in: not listed until it's back
        try:
            m.setdefault(lm["id"], model_hub.local_to_spec(lm))
        except Exception as e:
            print("[engine] bad local model entry:", lm.get("path"), e, flush=True)
    for um in model_hub.USER_MODELS:  # base models added from Hugging Face (Settings -> Models)
        try:
            m[um["id"]] = model_hub.to_spec(um)
        except Exception as e:
            print("[engine] bad added model entry:", um, e, flush=True)
    # 💾 Keep a copy on this Mac: a drive's file with a copy in Protoart's models folder uses the copy (loads in
    # seconds, not a minute). Its old id still works (settings, queued jobs, saved chains) but isn't listed twice.
    import local_copy
    for lm in list(model_hub.LOCAL_MODELS) + model_hub.scan_model_dirs():
        cid = lm.get("path") and local_copy.copy_id(lm["path"])
        if cid and cid in m and lm["id"] in m:
            m[cid] = dict(m[cid], kept_from=lm["path"])
            m[lm["id"]] = dict(m[cid], kept_as=cid)
    for mid, fam in model_hub.MODEL_FAMILY.items():  # Settings → Fix a type: you picked SD 1.5 / SDXL for a file
        spec = m.get(mid)
        if spec and fam in ("sd15", "sdxl") and spec.get("family") != fam and (spec.get("file") or spec.get("tdict")):
            m[mid] = dict(spec, family=fam, detected_family=spec.get("family"), family_set=True,
                          defaults=dict(model_hub.DEFAULTS[fam]),
                          min_ram_gb=16 if fam == "sdxl" else 8, low_memory_min_ram_gb=8 if fam == "sdxl" else None)
        elif spec and spec.get("family") == fam:
            m[mid] = dict(spec, family_set=True)
    _tell_apart(m)
    return m


def _tell_apart(m):
    """Two of your model files with the same name (e.g. photon_v1 in two folders) would look identical in the
    lists: add the folder each is in, and say when they're the same size (most likely the same file twice)."""
    from collections import defaultdict
    by_label = defaultdict(list)
    for k, v in m.items():
        if v.get("file") and ":" in k and not v.get("kept_as"):
            by_label[v["label"]].append(k)
    for label, keys in by_label.items():
        if len(keys) < 2:
            continue
        sizes = {}
        for k in keys:
            try:
                sizes[k] = os.path.getsize(m[k]["file"])
            except OSError:
                sizes[k] = None
        for k in keys:
            folder = os.path.basename(os.path.dirname(m[k]["file"])) or "/"
            same = sum(1 for o in keys if o != k and sizes[o] and sizes[o] == sizes[k]) > 0
            m[k] = dict(m[k], label=f"{label} · in “{folder}”" + (" · a copy" if same else ""),
                        duplicate_of=[o for o in keys if o != k])


class Engine:
    def wait_if_paused(self):
        """Pause: hold here (between steps) until Resume, Stop or Skip. Memory stays as it is, so it carries on
        exactly where it was."""
        if not self.pause_event.is_set():
            return
        print("[engine] paused", flush=True)
        while self.pause_event.is_set() and not self.cancel_event.is_set() and not self.skip_event.is_set():
            time.sleep(0.2)
        print("[engine] resumed", flush=True)

    def __init__(self, library, emit):
        self.library = library
        self._emit = emit  # sends an event to the UI
        self.cancel_event = threading.Event()
        self.skip_event = threading.Event()  # "skip this image": stop the current image, keep the job going
        self.pause_event = threading.Event()  # set = paused: the running image waits at its next step
        self.current_job = None  # id of the queue job being run (tags progress/image events)
        self.pipe = None
        self.loaded_key = None  # (model_id, use_pe)
        self.family = None
        self.base_scheduler_config = None
        self.adapters = {}  # adapter_id -> lora file
        self.fused_key = None    # LoRA mix currently baked into the weights
        self.fuse_ok = False     # plain fp16 SD/SDXL weights can have LoRAs fused
        self.embeddings = set()  # textual-inversion tokens loaded into the current model
        self.face_restorer = None  # RestoreFormer, small: kept loaded once used
        self.enhancer = None       # Finegrain enhancer, large: shares the "big model" slot
        self.interrogator = None   # Florence-2 / WD tagger for image -> prompt
        self.output_format = ("png", 92)  # (format, JPEG quality); set from Settings for each job
        self.prompt_enhancer = None
        self.i2i_pipe = None        # image-to-image pipeline sharing the loaded model's weights
        self.embed_cache = {}       # ERNIE lean mode: (repo, text) -> text embedding
        self._torch = None
        self.mlx = None             # fast ERNIE engine (separate process), started on demand
        self.moss = None            # MOSS-Image's engine (its own process), started on demand
        self.controlnets = {}       # ControlNet models by type (share the loaded SD 1.5 model)
        self.cn_pipes = {}          # (type, img2img) -> pipeline sharing the loaded model's weights
        self.depth_estimator = None
        self.inp_pipe = None        # inpainting pipeline sharing the loaded model's weights
        self.ip_key = None          # (model, kind) of the loaded image prompt adapter, or None
        self.face_finder = None     # RetinaFace for the face detailer
        self.face_swapper = None    # 🔁 Face swap (inswapper_128 + ArcFace, ONNX Runtime on the CPU)
        self.pose_detector = None   # person + body keypoints for the OpenPose ControlNet
        self.previews = True        # live previews while generating (Settings)
        self.nsfw_checker = None    # content filter's picture checker (small, CPU), loaded on demand
        self.check_pictures = False  # Settings → Content filter: check each new picture (set per job)

    def emit(self, msg):
        if self.current_job is not None:
            msg = dict(msg, job=self.current_job)
        self._emit(msg)

    # ------------------------------------------------------------ info
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def device(self):
        t = self.torch()
        return "mps" if t.backends.mps.is_available() else "cpu"

    def system_info(self):
        info = {"ram_gb": total_ram_gb(), "free_disk_gb": free_disk_gb(paths.HOME),
                "data_dir": str(paths.HOME), "loras_dir": str(paths.LORAS_DIR),
                "models_dir": str(paths.MODELS_DIR)}
        try:
            t = self.torch()
            import diffusers
            info.update(device=self.device(), torch=t.__version__, diffusers=diffusers.__version__,
                        ernie_supported=hasattr(diffusers, "ErnieImagePipeline"))
        except Exception as e:  # torch missing or broken install
            info.update(device="unavailable", error=str(e))
        return info

    def list_models(self):
        ram = total_ram_gb()
        return [
            {"id": k, "label": v["label"], "family": v["family"], "defaults": v["defaults"],
             "quant": v.get("quant"), "download_gb": v["download_gb"], "pe_gb": v.get("pe_gb"),
             "min_ram_gb": v["min_ram_gb"], "low_memory_min_ram_gb": v.get("low_memory_min_ram_gb"),
             "cached": self.spec_cached(v), "disk_gb": v.get("disk_gb"),
             "has_original": bool(v.get("local_name") and self.local_copy(v) and v.get("repo")
                                  and self._is_cached(v["repo"])),
             "enough_ram": ram == 0 or ram >= min(v["min_ram_gb"], v.get("low_memory_min_ram_gb") or 999),
             "auto_low_memory": bool(ram and v.get("low_memory_min_ram_gb") and ram < v["min_ram_gb"]),
             "sizes": ERNIE_SIZES if v["family"] == "ernie" else _moss_sizes() if v["family"] == "moss" else None,
             "source": v.get("source"),
             "engine": v.get("engine", "torch"), "fast_mode": v["family"] in FAST_LORAS and not v.get("engine"),
             "clip_skip": recommended_clip_skip(v, k), "flavor": model_flavor(v, k),
             "path": (str(v.get("file") or v.get("tdict") or v.get("local_dir") or "") or None) if ":" in k else None,
             # your own file that isn't where it was (moved, renamed, or its drive isn't connected): right-click → Reconnect
             "missing": bool(":" in k and (v.get("file") or v.get("tdict") or v.get("local_dir"))
                             and not os.path.exists(str(v.get("file") or v.get("tdict") or v.get("local_dir")))),
             "family_set": bool(v.get("family_set")), "detected_family": v.get("detected_family"),
             "single_file": bool(v.get("file") or v.get("tdict")) and ":" in k, "inpaint": bool(v.get("inpaint")),
             "nsfw_auto": nsfw_words.name_flag(v["label"], k, os.path.basename(
                 str(v.get("file") or v.get("tdict") or v.get("local_dir") or ""))),
             "kept_from": v.get("kept_from"),
             "can_keep": bool(":" in k and k.startswith("local:") and str(v.get("file") or "").lower().endswith(".safetensors")
                              and str(os.path.realpath(str(v.get("file")))).startswith("/Volumes/"))}
            for k, v in all_models().items() if not v.get("kept_as")
        ]

    def download_model(self, model_id):
        """Fetch a model's files so it's ready offline later, without loading it (Models tab -> Download).
        Works for any catalogue model, even ones too big for this Mac to run."""
        import threading as _th
        import net
        models = all_models()
        if model_id not in models:
            raise UserError(f"Unknown model: {model_id}")
        spec = models[model_id]
        if self.spec_cached(spec):
            self.emit({"type": "status", "msg": f"{spec['label']} is already on this Mac"})
            return {"model": model_id, "already": True}
        if net.strict:
            raise UserError("Strict offline mode is on (Settings → Privacy): turn it off to download models.")
        size = float(spec.get("download_gb") or 0)
        free = free_disk_gb(_hub_dir())
        if free < size + 3:
            raise UserError(f"{spec['label']} is a ~{size:.0f} GB download but only {free} GB is free.")
        from huggingface_hub import hf_hub_download, snapshot_download
        jobs = []
        if spec.get("family") == "anima":
            import anima
            turbo = bool(spec.get("anima_turbo"))
            jobs.append(lambda: snapshot_download(anima.DIFFUSERS_REPO, ignore_patterns=["transformer/*"] if turbo else None))
            if turbo:
                jobs.append(lambda: hf_hub_download(anima.WEIGHTS_REPO, anima.TURBO_FILE))
        elif spec.get("engine") == "mlx":
            jobs.append(lambda: snapshot_download(spec["mlx_repo"], revision=spec.get("mlx_revision"),
                                                      allow_patterns=spec.get("mlx_patterns")))
            if spec.get("moss_turbo"):  # MOSS-Image Turbo's two small adapters
                import moss_engine
                jobs.append(lambda: moss_engine.turbo_files(download=True))
        elif spec.get("gguf_file"):
            jobs.append(lambda: hf_hub_download(spec["gguf_repo"], spec["gguf_file"]))
            jobs.append(lambda: snapshot_download(spec["repo"], ignore_patterns=["transformer/*", "pe/*", "pe_tokenizer/*"]))
        elif spec.get("variant"):
            import diffusers
            skip = {"safety_checker": None, "feature_extractor": None} if spec["family"] == "sd15" else {}
            jobs.append(lambda: diffusers.DiffusionPipeline.download(spec["repo"], variant=spec["variant"],
                                                                     use_safetensors=True, **skip))
        elif spec.get("hub_file"):
            jobs.append(lambda: hf_hub_download(spec["repo"], spec["hub_file"]))
        elif spec.get("repo"):
            jobs.append(lambda: snapshot_download(spec["repo"], ignore_patterns=["pe/*", "pe_tokenizer/*"]))
        else:
            return {"model": model_id, "already": True}
        cache = paths.hub_cache()

        def cache_bytes():
            total = 0
            for root, _dirs, files in os.walk(cache):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return total

        start_bytes = cache_bytes()
        err = []
        done = _th.Event()

        def run():
            try:
                for j in jobs:
                    j()
            except Exception as e:  # reported below
                err.append(e)
            finally:
                done.set()

        net.set_online(True)
        try:
            self.emit({"type": "status", "msg": f"Downloading {spec['label']} (~{size:.1f} GB)…"})
            _th.Thread(target=run, daemon=True).start()
            total_mb = max(1, int(size * 1000))
            last_mb, last_change = -1, time.time()
            while not done.wait(1.5):
                if self.cancel_event.is_set():
                    # the download thread stops when the engine restarts; files already fetched are kept
                    raise Cancelled()
                got_mb = max(0, int((cache_bytes() - start_bytes) / 1e6))
                if got_mb != last_mb:
                    last_mb, last_change = got_mb, time.time()
                elif time.time() - last_change > 300:  # nothing arrived for 5 minutes: don't sit there forever
                    raise UserError(f"The download of {spec['label']} stalled (nothing arrived for 5 minutes). "
                                    "Check the internet connection, then press Download again: it carries on "
                                    "from where it stopped. If it stalls again, use Tools → Force stop & restart first.")
                self.emit({"type": "progress", "image": 1, "images": 1, "step": min(got_mb, total_mb - 1), "steps": total_mb})
                self.emit({"type": "status", "msg": f"Downloading {spec['label']}: {got_mb / 1000:.1f} of ~{size:.1f} GB"})
        finally:
            net.set_online(False)
        if err:
            problem = net.hf_auth_problem(err[0], spec.get("label"))
            raise UserError(problem or f"Couldn't download {spec['label']}: {err[0]}")
        self.emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 1})
        self.emit({"type": "status", "msg": f"{spec['label']} downloaded: ready to use offline"})
        return {"model": model_id, "downloaded": True}

    def unload(self, keep_mlx=False):
        if not keep_mlx and self.mlx is not None:
            self.mlx.stop()  # the fast engine holds its own copy of a model in memory
        if self.moss is not None:
            self.moss.stop()
        self.pipe = None
        self.i2i_pipe = None
        self.inp_pipe = None
        self.ip_key = None
        self.loaded_key = None
        self.adapters = {}
        self.fused_key = None
        self.embeddings = set()
        self.controlnets = {}
        self.cn_pipes = {}
        gc.collect()
        try:
            t = self.torch()
            if hasattr(t, "mps") and t.backends.mps.is_available():
                t.mps.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _is_cached(repo, with_pe=False, ignore_extra=(), revision=None):
        """True if the model files are already downloaded (no network access)."""
        try:
            from huggingface_hub import snapshot_download
            ignore = list(ignore_extra) + ([] if with_pe else ["pe/*", "pe_tokenizer/*"])
            snapshot_download(repo, local_files_only=True, ignore_patterns=ignore or None, revision=revision)
            return True
        except Exception:
            return False

    @staticmethod
    def local_copy(spec):
        name = spec.get("local_name")
        if not name:
            return None
        d = paths.QUANT_DIR / name
        return d if (d / "model_index.json").exists() else None

    def spec_cached(self, spec, with_pe=False):
        if spec.get("family") == "anima":
            import anima
            return anima.is_cached(bool(spec.get("anima_turbo")))
        if self.local_copy(spec):
            return True
        if spec.get("file") or spec.get("tdict") or spec.get("local_dir"):
            return True
        if spec.get("moss"):
            import moss_engine
            return bool(moss_engine.weights_dir()) and (not spec.get("moss_turbo") or bool(moss_engine.turbo_files()))
        if spec.get("engine") == "mlx":
            if spec.get("mlx_patterns"):
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(spec["mlx_repo"], local_files_only=True, allow_patterns=spec["mlx_patterns"])
                    return True
                except Exception:
                    return False
            return self._is_cached(spec["mlx_repo"], revision=spec.get("mlx_revision"))
        if spec.get("hub_file"):
            from huggingface_hub import try_to_load_from_cache
            return isinstance(try_to_load_from_cache(spec["repo"], spec["hub_file"]), str)
        if spec.get("variant"):
            # SD/SDXL repos contain many alternative files; only the chosen variant needs to be present.
            try:
                import diffusers
                skip = {"safety_checker": None, "feature_extractor": None} if spec["family"] == "sd15" else {}
                diffusers.DiffusionPipeline.download(spec["repo"], variant=spec["variant"], use_safetensors=True,
                                                     local_files_only=True, **skip)
                return True
            except Exception:
                return False
        if spec.get("gguf_file"):
            from huggingface_hub import try_to_load_from_cache
            if not isinstance(try_to_load_from_cache(spec["gguf_repo"], spec["gguf_file"]), str):
                return False
            return self._is_cached(spec["repo"], with_pe, ignore_extra=["transformer/*"])
        return self._is_cached(spec["repo"], with_pe) if spec.get("repo") else True

    @staticmethod
    def is_lean(spec):
        """Lean mode (Macs under 20 GB): ERNIE's text encoder is loaded only while encoding the prompt,
        then freed before the image model runs, similar to how Draw Things keeps memory low."""
        ram = total_ram_gb()
        return spec["family"] == "ernie" and 0 < ram < 20

    def _file_unet_8bit(self, spec, dtype, rcp):
        """The 8-bit UNet of a checkpoint file (Pony, Illustrious, your own SDXL files) for Low-memory mode.
        Loading a file straight into 8-bit leaves some weights empty (the "meta tensor" error), so the UNet is
        loaded normally, converted, and the 8-bit copy is saved: the next loads are quick and light."""
        import diffusers
        import sdnq
        src = Path(spec["file"])
        st = src.stat()
        dest = paths.QUANT_DIR / f"unet8_{_slug(src.stem)}_{st.st_size}_{int(st.st_mtime)}"
        if (dest / "quantization_config.json").exists():
            try:
                return sdnq.load_sdnq_model(str(dest), model_cls=diffusers.UNet2DConditionModel, dtype=dtype)
            except Exception as e:  # damaged copy: make it again
                print("[engine] saved 8-bit copy unreadable, converting again:", e, flush=True)
                shutil.rmtree(dest, ignore_errors=True)
        self.emit({"type": "status", "msg": f"Converting {spec['label']} to 8-bit (first time only, a few minutes)…"})
        unet = diffusers.UNet2DConditionModel.from_single_file(
            str(src), torch_dtype=dtype, config=rcp, subfolder="unet", local_files_only=True)
        unet = sdnq.sdnq_post_load_quant(unet, weights_dtype="int8", torch_dtype=dtype)
        gc.collect()
        if free_disk_gb(paths.QUANT_DIR.parent) > 6:
            try:
                self.emit({"type": "status", "msg": "Saving the 8-bit copy (~2.6 GB) so it loads quickly next time…"})
                dest.parent.mkdir(parents=True, exist_ok=True)
                sdnq.save_sdnq_model(unet, str(dest))
            except Exception as e:  # still usable now; converted again next time
                shutil.rmtree(dest, ignore_errors=True)
                print("[engine] couldn't save the 8-bit copy:", e, flush=True)
        return unet

    def load(self, model_id, use_pe=False, force=False, low_memory=False):
        models = all_models()
        if model_id not in models:
            raise UserError(f"Unknown model: {model_id}")
        spec = models[model_id]
        own = spec.get("file") or spec.get("tdict") or spec.get("local_dir")
        if own and not os.path.exists(str(own)):
            raise UserError(f"“{spec['label']}” isn't where it was ({own}): it was moved or renamed, or its drive isn't "
                            "connected. Right-click it in the model list → Reconnect… to point the app at the file.")
        if spec.get("local_dir"):  # a diffusers folder on this Mac: load it like a repo, never online
            spec = dict(spec, repo=spec["local_dir"])
        family = spec["family"]
        use_pe = bool(use_pe) and family == "ernie"
        # Low-memory (8-bit) only for SDXL: SD 1.5 already fits in 8 GB, and 8-bit loading of single-file
        # SD 1.5 models was a source of loading errors. ERNIE builds are already quantized.
        low_memory = bool(low_memory) and family == "sdxl"
        ram0 = total_ram_gb()
        if family == "sdxl" and spec.get("low_memory_min_ram_gb") and 0 < ram0 < spec["min_ram_gb"]:
            # SDXL on a Mac with less memory than it needs (8 GB): always the 8-bit UNet. Half the memory, practically
            # the same pictures, and no swapping to disk (full precision took minutes per step there)
            low_memory = True
        if spec.get("quantize_unet"):
            low_memory = False  # already 8-bit
        lean = self.is_lean(spec)
        use_pe = False  # the prompt enhancer now runs as its own step (see enhance_prompt)
        key = (model_id, use_pe, low_memory, lean)
        if self.loaded_key == key and self.pipe is not None:
            return

        need_ram = spec["min_ram_gb"] + (PE_EXTRA_RAM_GB if use_pe else 0)
        if low_memory:
            need_ram = spec.get("low_memory_min_ram_gb") or max(4, need_ram - LOW_MEMORY_SAVES_GB.get(family, 0))
        ram = total_ram_gb()
        if lean:
            need_ram = max(8, need_ram - 6)  # text encoder isn't resident during generation
        if ram and ram < need_ram * 0.5 and not force:
            raise UserError(f"{spec['label']} needs about {need_ram} GB of memory and this Mac has {ram} GB. "
                            "Turn on 'Load anyway' in Settings to try.")
        if ram and ram < need_ram + 2:
            # Advisory only: macOS can page memory to disk, so it usually still works, just slower.
            self.emit({"type": "status", "msg": f"Heads up: {ram} GB is tight for {spec['label']}; it may be slow."})

        repo = spec.get("repo")
        if repo and not self.spec_cached(spec, with_pe=use_pe):
            size = spec["download_gb"] + ((spec.get("pe_gb") or 0) if use_pe else 0)
            free = free_disk_gb(_hub_dir())
            if free < size + 5:
                raise UserError(
                    f"{spec['label']} needs a ~{size:.0f} GB download but only {free} GB is free. "
                    "Free up disk space first (keep 5 GB spare).")
            self.emit({"type": "status", "msg": f"Downloading {spec['label']} (first time only, ~{size:.0f} GB)…"})
        else:
            self.emit({"type": "status", "msg": f"Loading {spec['label']}…"})

        load_started = time.time()
        self.unload()
        self.enhancer = None  # free the Finegrain enhancer's memory too
        if 0 < ram < 16:
            self.drop_helpers()  # smaller Macs: only the image model stays in memory
        t = self.torch()
        import diffusers
        device = self.device()

        if family == "anima":
            import anima
            turbo = bool(spec.get("anima_turbo"))
            if not anima.is_cached(turbo):
                free = free_disk_gb(_hub_dir())
                if free < spec["download_gb"] + 5:
                    raise UserError(f"{spec['label']} needs a ~{spec['download_gb']:.0f} GB download but only {free} GB is free.")
                self.emit({"type": "status", "msg": f"Downloading {spec['label']} (first time only, ~{spec['download_gb']:.0f} GB)…"})
            dtype = t.bfloat16 if device == "mps" else t.float32
            self.pipe = anima.load_pipeline(turbo, dtype, device, emit=self.emit)
            if 0 < ram < 16:
                anima.lean_memory(self.pipe, device)  # only the working part in GPU memory
            self.family = "anima"
            self.loaded_key = key
            print(f"[timing] loaded {model_id} in {time.time() - load_started:.1f}s", flush=True)
            self.emit({"type": "status", "msg": f"{spec['label']} loaded"})
            return
        if spec.get("hub_file"):  # single checkpoint file from Hugging Face: download, then load like a local file
            from huggingface_hub import hf_hub_download
            spec = dict(spec, file=hf_hub_download(spec["repo"], spec["hub_file"]))
        if family == "other":
            # A newer model type added from Hugging Face: let diffusers pick the right pipeline.
            dtype = t.bfloat16 if device == "mps" else t.float32
            kw = {"torch_dtype": dtype}
            if spec.get("variant"):
                kw["variant"] = spec["variant"]
            if spec.get("local_dir"):
                kw["local_files_only"] = True
            try:
                pipe = diffusers.AutoPipelineForText2Image.from_pretrained(repo, **kw)
            except Exception as auto_err:
                try:
                    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, **kw)
                except Exception:
                    raise UserError(f"This diffusers version can't load {spec['label']} ({spec.get('class_name')}): "
                                    f"{auto_err}. Re-running setup.sh updates diffusers; very new models can need a "
                                    "newer version than is released.")
        elif family == "ernie":
            if not hasattr(diffusers, "ErnieImagePipeline"):
                raise UserError("This diffusers version has no ERNIE support. Re-run setup.sh to update it.")
            if spec.get("quant"):
                try:
                    import sdnq  # noqa: F401  registers the SDNQ format with diffusers/transformers
                except ImportError:
                    raise UserError("Quantized ERNIE models need the 'sdnq' package. Re-run setup.sh.")
            # Passing pe=None also stops diffusers from downloading the prompt enhancer.
            extra = {} if use_pe else {"pe": None, "pe_tokenizer": None}
            if lean:
                extra["text_encoder"] = None  # loaded on demand in _ernie_embeddings
            if spec.get("gguf_file"):
                from huggingface_hub import hf_hub_download
                self.emit({"type": "status", "msg": f"Loading 6-bit transformer ({spec['gguf_file']})…"})
                gguf_path = hf_hub_download(spec["gguf_repo"], spec["gguf_file"])
                extra["transformer"] = diffusers.ErnieImageTransformer2DModel.from_single_file(
                    gguf_path, config=spec["config_repo"], subfolder="transformer",
                    quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=t.bfloat16),
                    torch_dtype=t.bfloat16)
            if spec.get("local_dir"):
                extra["local_files_only"] = True
            pipe = diffusers.ErnieImagePipeline.from_pretrained(repo, torch_dtype=t.bfloat16, **extra)
        else:
            dtype = t.float16 if device == "mps" else t.float32
            cls = diffusers.StableDiffusionXLPipeline if family == "sdxl" else diffusers.StableDiffusionPipeline
            if spec.get("inpaint"):
                # an inpainting checkpoint: 9-channel UNet, loaded as an inpainting pipeline with a matching recipe
                cls = diffusers.StableDiffusionXLInpaintPipeline if family == "sdxl" else diffusers.StableDiffusionInpaintPipeline
            kw = {"torch_dtype": dtype}
            if low_memory:
                # Quantize the UNet to 8-bit while loading: roughly halves its memory use.
                # Disk usage is unchanged (the normal fp16 files are what's downloaded).
                from diffusers import PipelineQuantizationConfig, SDNQConfig
                kw["quantization_config"] = PipelineQuantizationConfig(
                    quant_mapping={"unet": SDNQConfig(weights_dtype="int8")})
            if spec.get("tdict"):
                self.emit({"type": "status", "msg": f"Loading {spec['label']} from DiffusionBee…"})
                import local_configs
                rcp = (local_configs.ensure_inpaint if spec.get("inpaint") else local_configs.ensure)(family)  # offline recipe
                tkw = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False,
                       "config": rcp, "local_files_only": True}
                if low_memory:
                    from diffusers import SDNQConfig
                    import tdict_reader
                    tkw["unet"] = diffusers.UNet2DConditionModel.from_single_file(
                        tdict_reader.TDict(spec["tdict"]).checkpoint_state_dict(), torch_dtype=dtype,
                        config=rcp, subfolder="unet", local_files_only=True,
                        quantization_config=SDNQConfig(weights_dtype="int8"))
                pipe = _from_tdict(cls, spec["tdict"], **tkw)
            elif spec.get("file"):
                if not os.path.isfile(spec["file"]):
                    raise UserError(f"The model file isn't there any more: {spec['file']}. It may have been moved, renamed "
                                    "or deleted (or it's on a drive that isn't connected). Models tab → ↻ Refresh updates "
                                    "the list.")
                import local_configs
                # your own checkpoint: load fully offline (an inpainting one with its own 9-channel recipe)
                rcp = (local_configs.ensure_inpaint if spec.get("inpaint") else local_configs.ensure)(family)
                fkw = {"config": rcp, "local_files_only": True}
                if family == "sd15":
                    fkw.update(safety_checker=None, requires_safety_checker=False)
                if low_memory:
                    unet = self._file_unet_8bit(spec, dtype, rcp)
                    pipe = cls.from_single_file(spec["file"], torch_dtype=dtype, unet=unet, **fkw)
                else:
                    pipe = cls.from_single_file(spec["file"], torch_dtype=dtype, **fkw)
            elif spec.get("sdnq") and spec.get("local_dir"):  # a smaller copy made in Tools → Quantize
                import sdnq
                unet = sdnq.load_sdnq_model(str(Path(spec["local_dir"]) / "unet"), model_cls=diffusers.UNet2DConditionModel,
                                            dtype=dtype)
                pipe = cls.from_pretrained(spec["local_dir"], unet=unet, torch_dtype=dtype, local_files_only=True,
                                           **({"safety_checker": None, "requires_safety_checker": False} if family == "sd15" else {}))
            elif self.local_copy(spec):
                import sdnq  # noqa: F401  needed to read the saved 8-bit weights
                pipe = cls.from_pretrained(str(self.local_copy(spec)), torch_dtype=dtype)
            else:
                kw["use_safetensors"] = True
                if spec.get("local_dir"):
                    kw["local_files_only"] = True
                if spec.get("variant"):
                    kw["variant"] = spec["variant"]
                if family == "sd15":
                    kw.update(safety_checker=None, requires_safety_checker=False)
                if spec.get("quantize_unet"):
                    from diffusers import PipelineQuantizationConfig, SDNQConfig
                    self.emit({"type": "status", "msg": f"Converting {spec['label']} to 8-bit (first time only)…"})
                    kw["quantization_config"] = PipelineQuantizationConfig(
                        quant_mapping={"unet": SDNQConfig(weights_dtype=spec["quantize_unet"])})
                pipe = cls.from_pretrained(repo, **kw)
                if spec.get("local_name"):
                    dest = paths.QUANT_DIR / spec["local_name"]
                    try:
                        self.emit({"type": "status", "msg": f"Saving the 8-bit copy ({spec.get('disk_gb', '?')} GB)…"})
                        pipe.save_pretrained(str(dest))
                    except Exception as err:  # keep going; we'll just convert again next time
                        shutil.rmtree(dest, ignore_errors=True)
                        print("[engine] couldn't save quantized copy:", err, flush=True)
                    else:
                        if spec.get("quantize_unet") and 0 < ram < 24:
                            # The conversion leaves the full-size original's memory behind for a while; on smaller
                            # Macs that made the first pictures crawl (the Mac swapped to disk). Start again from
                            # the compact copy just saved: exactly what every later load uses.
                            del pipe
                            gc.collect()
                            try:
                                if t.backends.mps.is_available():
                                    t.mps.empty_cache()
                            except Exception:
                                pass
                            import sdnq  # noqa: F401  needed to read the saved 8-bit weights
                            self.emit({"type": "status", "msg": f"Loading the 8-bit copy of {spec['label']}…"})
                            pipe = cls.from_pretrained(str(dest), torch_dtype=dtype)

        _repair_meta_tensors(pipe)  # a weight missing from a checkpoint file would otherwise crash .to(device)
        pipe = pipe.to(device)
        self.fuse_ok = family in ("sd15", "sdxl") and not low_memory and not spec.get("quantize_unet") and not spec.get("sdnq") \
            and not self.local_copy(spec)
        self._tuning = None  # memory/speed switches are set per job (see _tune_for_size)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self.pipe = pipe
        self.family = family
        self.loaded_key = key
        self.base_scheduler_config = dict(pipe.scheduler.config)
        print(f"[timing] loaded {model_id} in {time.time() - load_started:.1f}s "
              f"(lean={lean}, low_memory={low_memory}, ram={ram} GB, free {_free_ram_gb():.1f} GB, "
              f"swap {_swap_used_gb():.1f} GB)", flush=True)
        self.emit({"type": "status", "msg": f"{spec['label']} ready"})

    # ------------------------------------------------------------ training
    def train_lora(self, args):
        import trainer
        if self.mlx is not None:
            self.mlx.stop()
        self.unload()  # start from a clean model (no fused LoRAs left in its weights) and free memory
        self.drop_helpers()
        self._room_to_train()
        try:
            return trainer.train(self, args, self.emit, self.cancel_event, paths.LORAS_DIR)
        except trainer._Cancelled:
            raise Cancelled()
        except trainer.TrainError as e:
            raise UserError(str(e))
        finally:
            self.unload()  # training modifies the loaded model; always start fresh afterwards

    def _room_to_train(self, need_gb=1.5):
        """Training holds a model and its working memory for many minutes. On a Mac under 16 GB with too little
        memory free it would push macOS into heavy swapping, which can freeze the whole Mac: say so instead."""
        if not (0 < total_ram_gb() < 16):
            return
        gc.collect()
        free, swap = _free_ram_gb(), _swap_used_gb()
        if 0 <= free < need_gb and swap > 2:
            raise UserError(f"Only {free:.1f} GB of memory is free and {swap:.1f} GB is already on the disk, so "
                            "starting training now could make the whole Mac freeze. Quit the apps you aren't "
                            "using (web browsers, Discord, Claude, other image apps), then press Train again.")

    def train_embedding(self, args):
        """Train an embedding (textual inversion) on SD 1.5: Train tab → Embedding."""
        import embeddings
        import ti_trainer
        import trainer
        if self.mlx is not None:
            self.mlx.stop()
        self.unload()
        self.drop_helpers()
        self._room_to_train()
        try:
            return ti_trainer.train(self, args, self.emit, self.cancel_event, embeddings.own_dir())
        except trainer._Cancelled:
            raise Cancelled()
        except (ti_trainer.TIError, trainer.TrainError) as e:
            raise UserError(str(e))
        finally:
            self.unload()

    def train_autocaption(self, args):
        """Write .txt captions for training images that don't have one (or all, if overwrite)."""
        import trainer
        items = trainer.scan_dataset(args.get("folder"))
        todo = [it for it in items if args.get("overwrite") or not it["has_caption"]]
        mode = "tags" if args.get("mode") == "tags" else "describe"
        if 0 < total_ram_gb() < 16:
            self.unload()
        done = 0
        for i, it in enumerate(todo):
            if self.cancel_event.is_set():
                raise Cancelled()
            self.emit({"type": "status", "msg": f"Captioning {i + 1}/{len(todo)}: {it['name']}…"})
            self.emit({"type": "progress", "image": 1, "images": 1, "step": i + 1, "steps": len(todo)})
            res = self.interrogate({"path": it["path"], "mode": mode, "detail": "detailed", "_keep": True})
            it["caption"] = res.get("prompt", "")
            trainer.save_captions([it])
            done += 1
        return {"captioned": done, "items": trainer.scan_dataset(args.get("folder"), include_skipped=True)}

    # ------------------------------------------------------------ ControlNet
    def _get_depth_estimator(self, size="small"):
        """Depth Anything V2 (small / base / large); one size kept in memory at a time."""
        import controlnet as cnet
        size = cnet.depth_size(size)
        held = self.depth_estimator
        if not (isinstance(held, tuple) and held[0] == size):
            self.depth_estimator = None
            self.release_memory()
            repo, gb, _ = cnet.DEPTH_MODELS[size]
            if size == "large" and 0 < total_ram_gb() < 16 and self.pipe is not None:
                self.unload()  # 8 GB Macs: make room for the large depth model
            self.emit({"type": "status", "msg": f"Loading Depth Anything V2 {size.capitalize()}"
                                               + ("" if self._is_cached(repo) else f" (first time ~{gb:g} GB)") + "…"})
            self.depth_estimator = (size, cnet.load_depth_estimator(self.device(), size))
        return self.depth_estimator[1]

    def control_map(self, kind, image, width, height, already_map=False, depth_model="small"):
        import controlnet as cnet
        mk = cnet.guess_map(kind[len("user:"):]) if str(kind).startswith("user:") else cnet.map_kind(kind)  # your own: map from its name
        depth = self._get_depth_estimator(depth_model) if (mk == "depth" and not already_map) else None
        posed = self._get_pose_detector() if (mk == "openpose" and not already_map) else None
        try:
            return cnet.make_map(kind, image, width, height, depth_estimator=depth, already_map=already_map,
                                 pose_detector=posed)
        except ValueError as e:
            msg = str(e)
            raise UserError(msg[:1].upper() + msg[1:] + ".")

    def _controlnet_pipe(self, control, control_src, img2img, width, height):
        import controlnet as cnet
        import diffusers
        kind = control["type"]
        cmap = self.control_map(kind, control_src, width, height, bool(control.get("already_map")),
                                control.get("depth_model") or "small")
        if 0 < total_ram_gb() < 16 and (self.depth_estimator is not None or self.pose_detector is not None):
            # 8 GB Macs: the map is made; the depth / pose model would only crowd the image model while it draws
            self.depth_estimator = None
            self.pose_detector = None
            self.release_memory()
        union = cnet.is_union(kind)
        ck = "union" if union else kind  # one Union model serves every Union map type
        if ck not in self.controlnets:
            if 0 < total_ram_gb() < 16:
                self.controlnets = {}  # 8 GB Macs: keep only one ControlNet loaded
                self.cn_pipes = {}
                self.release_memory()
            dtype = self.pipe.unet.dtype
            self.controlnets[ck] = cnet.load(kind, self.pipe.unet, dtype, emit=self.emit).to(self.device())
            self.release_memory()  # hand back what loading it used, before the first step
        key = (ck, img2img)
        if key not in self.cn_pipes:
            if union:
                cls = (diffusers.StableDiffusionXLControlNetUnionImg2ImgPipeline if img2img
                       else diffusers.StableDiffusionXLControlNetUnionPipeline)
            elif self.family == "sdxl":
                cls = diffusers.StableDiffusionXLControlNetImg2ImgPipeline if img2img else diffusers.StableDiffusionXLControlNetPipeline
            else:
                cls = diffusers.StableDiffusionControlNetImg2ImgPipeline if img2img else diffusers.StableDiffusionControlNetPipeline
            self.cn_pipes[key] = cls.from_pipe(self.pipe, controlnet=self.controlnets[ck], **_keep_dtype(self.pipe))
        return self.cn_pipes[key], cmap

    def control_preview(self, args):
        """Make just the control map (shown in the Image to image tab before generating)."""
        import base64
        import io
        from PIL import Image
        try:
            import pictures
            img = Image.open(pictures.prepare(args["path"])).convert("RGB")
        except Exception:
            raise UserError("Couldn't open that image.")
        w, h = img.size
        scale = 512 / max(w, h)
        w, h = max(64, int(w * scale) // 8 * 8), max(64, int(h * scale) // 8 * 8)
        m = self.control_map(args.get("type", "canny"), img, w, h, bool(args.get("already_map")),
                             args.get("depth_model") or "small")
        buf = io.BytesIO()
        m.save(buf, "JPEG", quality=85)
        return {"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}

    def _emit_preview(self, latents, image_no):
        """Send a small live preview of the image being made (see preview.py)."""
        if latents is None:
            return
        import preview
        if self.family not in preview.FACTORS:
            return  # no quick preview for this model type
        try:
            lat = latents[0].detach().float().cpu().numpy()
            self.emit({"type": "preview", "image": image_no, "src": preview.latent_to_data_url(lat, self.family)})
        except Exception as e:
            print("[engine] preview failed:", e, flush=True)
            self.previews = False  # don't keep failing every step

    def _fast_lora_file(self, family):
        repo, fname, gb = FAST_LORAS[family]
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        cached = try_to_load_from_cache(repo, fname)
        if isinstance(cached, str):
            return cached
        if free_disk_gb(_hub_dir()) < gb + 1:
            raise UserError("Fast mode needs a small download but the disk is almost full.")
        self.emit({"type": "status", "msg": f"Downloading the fast-mode LoRA (first time only, ~{int(gb * 1000)} MB)…"})
        return hf_hub_download(repo, fname)

    @staticmethod
    def _region_place(box):
        """Where an area is, in words: "on the left", "at the top right", "in the middle"."""
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        h = "left" if cx < 0.36 else "right" if cx > 0.64 else ""
        v = "top" if cy < 0.36 else "bottom" if cy > 0.64 else ""
        if v and h:
            return f"at the {v} {h}"
        if v:
            return f"at the {v}"
        return f"on the {h}" if h else "in the middle"

    def regions_by_parts(self, args, force=False):
        """🗺 Regional prompts on FLUX.1 / ERNIE (fast engine), which can't split one picture's prompt by area:
        1. the whole picture is drawn from the main prompt plus each area's prompt with its place in words
           ("on the left, a woman in a red coat…"), which these models follow well;
        2. each area is then redrawn on its own from its own prompt (image to image on that part, a little bigger
           than the area) and blended back in with soft edges."""
        import regions
        import tempfile
        from PIL import Image, ImageDraw, ImageFilter
        rg = args["regions"]
        family = all_models()[args["model"]]["family"]
        rps = [x.strip() for x in rg.get("prompts") or [] if isinstance(x, str) and x.strip()]
        boxes = regions.parse_layout(rg.get("layout") or "cols", rg.get("ratios") or "", len(rps))
        main = (args.get("prompt") or "").strip().rstrip(".")
        places = [self._region_place(b) for b in boxes]
        composed = ". ".join([main] if main else []) + ("." if main else "")
        composed = (composed + " " + " ".join(f"{pl[0].upper() + pl[1:]}, {rp.rstrip('.')}." for pl, rp in zip(places, rps))).strip()
        seed = args.get("seed")
        seed = int(seed) if seed not in (None, "", -1) else random.randint(0, 2**31 - 1)
        refine = max(0.0, min(0.8, float(rg.get("refine", 0.5))))
        started = time.time()
        n_steps = 1 + (len(rps) if refine > 0 else 0)
        self.emit({"type": "status", "msg": f"Regions (fast engine): 1 of {n_steps} · the whole picture, each area in its place…"})
        base = dict(args, prompt=composed, regions=None, num_images=1, seed=seed, enhance_prompt=False, _raw=True)
        first = self.generate(base, force=force)
        if not first:
            raise Cancelled()
        img = first[0]["image"].convert("RGB")
        W, H = img.size
        tmpdir = Path(tempfile.mkdtemp(prefix="regions_", dir=str(paths.HOME / "tmp") if (paths.HOME / "tmp").is_dir() else None))
        if refine > 0:
            for k, (box, rp) in enumerate(zip(boxes, rps)):
                if self.cancel_event.is_set():
                    raise Cancelled()
                self.emit({"type": "status", "msg": f"Regions (fast engine): {k + 2} of {n_steps} · redrawing area {k + 1} ({places[k]})…"})
                x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
                mx, my = max(16, (x1 - x0) // 8), max(16, (y1 - y0) // 8)   # a margin, so the edges blend
                cx0, cy0, cx1, cy1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
                cw, ch = cx1 - cx0, cy1 - cy0
                crop = img.crop((cx0, cy0, cx1, cy1))
                f = max(1.0, 512 / min(cw, ch))
                f = min(f, 1280 / max(cw, ch))
                tw, th = max(256, int(cw * f) // 16 * 16), max(256, int(ch * f) // 16 * 16)
                src = tmpdir / f"area{k}.png"
                crop.resize((tw, th), Image.LANCZOS).save(src)
                if family == "flux2":
                    # FLUX.2 Klein can't start from a picture the usual way: it edits it with words instead
                    back = self._klein_edit(crop.resize((tw, th), Image.LANCZOS),
                                            f"Make this show {rp.rstrip('.')}, filling the frame the same way, with the same "
                                            f"lighting, colours and style as the picture, and natural edges.",
                                            seed + k + 1, steps=args.get("steps"))
                else:
                    part = self.generate(dict(args, prompt=f"{rp.rstrip('.')}, {main}" if main else rp, regions=None, num_images=1,
                                              init_image=str(src), strength=refine, width=tw, height=th, seed=seed + k + 1,
                                              enhance_prompt=False, _raw=True), force=force)
                    if not part:
                        raise Cancelled()
                    back = part[0]["image"]
                back = back.convert("RGB").resize((cw, ch), Image.LANCZOS)
                # soft-edged mask: full inside the area, fading out across the margin
                mask = Image.new("L", (cw, ch), 0)
                ImageDraw.Draw(mask).rectangle((x0 - cx0 if cx0 else 0, y0 - cy0 if cy0 else 0,
                                                cw - (cx1 - x1 if cx1 < W else 0), ch - (cy1 - y1 if cy1 < H else 0)), fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(max(4, min(mx, my) / 2)))
                if family == "flux2" and refine < 0.8:  # Klein's edit is all-or-nothing: the slider sets how much shows
                    mask = mask.point(lambda v: int(v * min(1.0, refine / 0.6)))
                img.paste(back, (cx0, cy0), mask)
        for f_ in tmpdir.glob("*"):
            try:
                f_.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass
        spec = all_models()[args["model"]]
        s0 = first[0]["settings"]
        settings = dict(s0, prompt=composed, seed=seed, width=img.width, height=img.height,
                        seconds=round(time.time() - started, 1),
                        regions={"layout": rg.get("layout"), "ratios": rg.get("ratios"), "prompts": rps, "refine": refine,
                                 "method": "by parts (fast engine)"})
        if main and main != composed:
            settings["original_prompt"] = args.get("prompt")
        path = self._save(img, settings)
        record = self.library.add_image(path, settings, kind="generated")
        record["thumb"] = str(thumbs.make(record["id"], path, img) or "") or None
        record = self._people_check(self._content_check(record, img), img)
        self.emit({"type": "image", "record": record})
        return [record]

    def _generate_anima(self, args, model_id, spec, prompts, negatives, originals, width, height, steps, cfg, seeds,
                        init_image, strength, force=False):
        """Anima (diffusers modular pipeline): text to image, or image to image with a start picture."""
        import anima
        if args.get("loras"):
            self.emit({"type": "status", "msg": "LoRAs aren't used with Anima yet: making the picture without them."})
        self.load(model_id, force=force)
        pipe = self.pipe
        anima.set_guidance(pipe, cfg)
        # Anima drifts into suggestive pictures easily. Unless the prompt already has one of its rating tags, it
        # gets "safe" (and the rest go in the negative prompt), so pictures stay all-ages by default.
        prompts = [anima.with_rating(p) for p in prompts]
        negatives = [anima.safe_negative(p, n) for p, n in zip(prompts, negatives)]
        t = self.torch()
        results = []
        n = len(prompts)
        for i in range(n):
            self.wait_if_paused()
            if self.cancel_event.is_set():
                raise Cancelled()
            started = time.time()
            last_preview = [0.0]

            def on_step(step, _i=i):
                self.wait_if_paused()
                if self.cancel_event.is_set():
                    raise Cancelled()
                if self.skip_event.is_set():
                    raise Skipped()
                self.emit({"type": "progress", "image": _i + 1, "images": n, "step": min(step + 1, steps), "steps": steps})
                if step == 0 or (step + 1) % 5 == 0:
                    print(f"[timing] anima step {step + 1}/{steps}: {time.time() - started:.1f}s elapsed", flush=True)

            watch = anima.StepWatch(pipe.transformer, on_step)
            kw = dict(prompt=prompts[i], negative_prompt=negatives[i] or "", height=height, width=width,
                      num_inference_steps=steps, generator=t.Generator("cpu").manual_seed(seeds[i]), output_type="pil")
            if init_image is not None:
                kw.update(image=init_image.resize((width, height)), strength=strength)
            self.emit({"type": "status", "msg": f"Generating image {i + 1} of {n} (Anima)…"})
            self.skip_event.clear()
            try:
                out = pipe(**kw)
            except Skipped:
                continue
            finally:
                watch.remove()
            image = out.images[0] if hasattr(out, "images") else out.get("images")[0]
            secs = time.time() - started
            print(f"[timing] anima image {i + 1}/{n}: {secs:.1f}s for {steps} steps at {width}x{height}", flush=True)
            face_info = None
            if args.get("restore_faces"):
                image, face_info = self._restore_faces(image, args)
            settings = {"model": model_id, "prompt": prompts[i], "negative_prompt": negatives[i], "width": image.width,
                        "height": image.height, "steps": steps, "cfg": cfg, "sampler": "flow_euler", "seed": seeds[i],
                        "seconds": round(secs, 1), "loras": [], "licence": anima.LICENCE}
            if originals[i] != prompts[i]:
                settings["original_prompt"] = originals[i]
            if init_image is not None:
                settings.update(init_image=str(args["init_image"]), strength=strength)
            if face_info:
                settings["face_restore"] = face_info
            path = self._save(image, settings)
            record = self.library.add_image(path, settings, kind="img2img" if init_image is not None else "generated")
            record["thumb"] = str(thumbs.make(record["id"], path, image) or "") or None
            record = self._people_check(self._content_check(record, image), image)
            results.append(record)
            self.emit({"type": "image", "record": record})
        self.release_memory()
        return results

    def _generate_mlx(self, args, model_id, spec, prompt, negative, width, height, steps, cfg, n, seed,
                      original_prompt, force=False):
        """ERNIE and FLUX on the fast engine (MLX worker process)."""
        import mlx_engine
        mlx_init, mlx_strength = None, None
        if args.get("init_image"):
            if spec["family"] not in ("flux", "ernie"):
                raise UserError("FLUX.2 Klein changes a picture with ✨ Edit with words (Image to image): pick it there.")
            import pictures
            from PIL import Image as _Im
            src = _Im.open(pictures.prepare(args["init_image"])).convert("RGB").resize((width, height), _Im.LANCZOS)
            tmp = paths.HOME / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            mlx_init = str(tmp / f"mlx_init_{int(time.time() * 1000)}.png")
            src.save(mlx_init)
            # mflux's image strength is how much of the picture to keep: the opposite of "change strength"
            mlx_strength = max(0.05, min(0.95, 1.0 - float(args.get("strength", 0.6))))
        from PIL import Image
        fam = spec["family"]
        fam_name = {"ernie": "ERNIE", "flux": "FLUX", "flux2": "FLUX.2"}.get(fam, fam)
        need = spec.get("min_ram_gb") or 8
        ram0 = total_ram_gb()
        if ram0 and ram0 < need * 0.5 and not force:
            raise UserError(f"{spec['label']} needs about {need} GB of memory and this Mac has {ram0} GB. "
                            "Turn on 'Load anyway' in Settings to try.")
        if ram0 and ram0 < need:
            self.emit({"type": "status", "msg": f"Heads up: {ram0} GB is tight for {spec['label']}; it may be slow."})
        try:
            ram = self._mlx_load(spec, args.get("loras") or [])
            self.loaded_key = None
            self.family = fam
            out_dir = paths.HOME / "tmp"
            out_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for i in range(n):
                if self.cancel_event.is_set():
                    raise Cancelled()
                img_seed = (seed + i) % 2**32
                # per-image prompts (wildcards) arrive as lists
                p_i = prompt[i] if isinstance(prompt, list) else prompt
                n_i = negative[i] if isinstance(negative, list) else negative
                o_i = original_prompt[i] if isinstance(original_prompt, list) else original_prompt
                out = out_dir / f"mlx_{os.getpid()}_{i}.png"
                self.emit({"type": "status", "msg": f"Generating image {i + 1} of {n} (fast engine)…"})
                started = time.time()

                def on_event(ev, _i=i):
                    if ev.get("type") == "progress":
                        self.emit({"type": "progress", "image": _i + 1, "images": n,
                                   "step": ev["step"], "steps": ev["steps"]})
                    elif ev.get("type") == "preview":
                        self.emit({"type": "preview", "image": _i + 1, "src": ev["src"]})
                    elif ev.get("type") == "status":
                        self.emit(ev)

                self.skip_event.clear()
                try:
                    req = {"cmd": "generate", "prompt": p_i, "negative_prompt": n_i,
                           "width": width, "height": height, "steps": steps, "cfg": cfg, "seed": img_seed,
                           "out": str(out), "preview": self.previews}
                    if mlx_init:  # start from a picture (model chain, image to image): FLUX.1 and ERNIE
                        req.update(init_image=mlx_init, image_strength=mlx_strength)
                    self.mlx.request(req, on_event=on_event,
                                     skip_event=self.skip_event)
                except mlx_engine.CancelledError:
                    if self.skip_event.is_set() and not self.cancel_event.is_set():
                        self.skip_event.clear()
                        self.emit({"type": "status", "msg": f"Skipped image {i + 1} of {n}"})
                        self.emit({"type": "skipped", "image": i + 1, "images": n})
                        continue
                    raise
                image = Image.open(out).convert("RGB")
                try:
                    out.unlink()
                except OSError:
                    pass
                secs = time.time() - started
                print(f"[timing] fast engine image {i + 1}/{n}: {secs:.1f}s for {steps} steps at {width}x{height} "
                      f"({secs / max(1, steps):.2f}s/step)", flush=True)
                face_info = None
                if args.get("restore_faces"):
                    image, face_info = self._restore_faces(image, args)
                settings = {
                    "model": model_id, "prompt": p_i, "negative_prompt": n_i,
                    "width": image.width, "height": image.height, "steps": steps, "cfg": cfg,
                    "sampler": "flow_euler", "seed": img_seed, "engine": "mlx",
                    "loras": [{"name": l["name"], "weight": float(l.get("weight", 1.0))} for l in args.get("loras") or []],
                    "seconds": round(secs, 1),
                }
                if o_i and o_i != p_i:
                    settings.update(original_prompt=o_i)
                    if args.get("enhance_prompt"):
                        settings["enhance_prompt"] = True
                if face_info:
                    settings["face_restore"] = face_info
                if args.get("_raw"):  # a step of a bigger job (regions on the fast engine): not saved on its own
                    results.append({"image": image, "seed": img_seed, "seconds": round(secs, 1), "settings": settings})
                    continue
                path = self._save(image, settings)
                record = self.library.add_image(path, settings, kind="generated")
                record["thumb"] = str(thumbs.make(record["id"], path, image) or "") or None
                record = self._people_check(self._content_check(record, image), image)
                results.append(record)
                self.emit({"type": "image", "record": record})
            return results
        except mlx_engine.CancelledError:
            raise Cancelled()
        except mlx_engine.EngineError as e:
            raise UserError(str(e))

    def _generate_moss(self, args, model_id, spec, prompt, negative, width, height, steps, cfg, n, seed,
                       original_prompt, force=False):
        """MOSS-Image (its own worker process; see moss_engine.py). Text to image only."""
        import moss_engine
        import net
        from PIL import Image
        if args.get("init_image"):
            raise UserError("MOSS-Image makes new pictures from a prompt only: for Image to image, pick an SD 1.5 / SDXL "
                            "model or FLUX.2 Klein (Edit with words).")
        if args.get("loras"):
            raise UserError("MOSS-Image can't use LoRAs in this version: take them out of the LoRA list first.")
        need = spec.get("min_ram_gb") or 16
        ram = total_ram_gb()
        if ram and ram < 8 and not force:
            raise UserError(f"{spec['label']} needs about {need} GB of memory and this Mac has {ram} GB.")
        if ram and ram < need:
            self.emit({"type": "status", "msg": f"Heads up: {ram:.0f} GB is tight for MOSS-Image: expect it to be slow, "
                                                "and close other apps."})
        # only one big model in memory at a time
        self.unload()
        self.enhancer = None
        self.drop_helpers()
        if self.moss is None:
            self.moss = moss_engine.MossEngine(self.emit, self.cancel_event, self.pause_event)
        try:
            weights = moss_engine.weights_dir()
            adapters = moss_engine.turbo_files() if spec.get("moss_turbo") else []
            if not weights or adapters is None or not moss_engine.installed():
                if not net.is_online():  # the backend tries again with the connection open (unless Strict offline)
                    raise UserError("offline: MOSS-Image isn't on this Mac yet")
                need_gb = (spec["download_gb"] if not weights else 0) + (spec.get("engine_gb", 1) if not moss_engine.installed() else 0)
                free = free_disk_gb(_hub_dir())
                if free < need_gb + 3:
                    raise UserError(f"{spec['label']} needs about {need_gb:.0f} GB of downloads but only {free} GB is free.")
            if not weights or adapters is None:
                self.emit({"type": "status", "msg": f"Downloading MOSS-Image (first time only, ~{spec['download_gb']:.0f} GB)…"})
                weights = moss_engine.weights_dir(download=True)
                if spec.get("moss_turbo"):
                    adapters = moss_engine.turbo_files(download=True)
            if not moss_engine.installed():
                self.moss.install()
            if self.cancel_event.is_set():
                raise Cancelled()
            cache = paths.HOME / "moss-prompts"
            cache.mkdir(parents=True, exist_ok=True)
            self.moss.request({"cmd": "load", "dir": weights, "turbo": bool(spec.get("moss_turbo")),
                               "adapters": adapters or [], "cache": str(cache)},
                              on_event=lambda ev: ev.get("type") == "status" and self.emit(ev))
            self.loaded_key = None
            self.family = "moss"
            out_dir = paths.HOME / "tmp"
            out_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for i in range(n):
                if self.cancel_event.is_set():
                    raise Cancelled()
                img_seed = (seed + i) % 2**31
                p_i = prompt[i] if isinstance(prompt, list) else prompt
                n_i = negative[i] if isinstance(negative, list) else negative
                o_i = original_prompt[i] if isinstance(original_prompt, list) else original_prompt
                out = out_dir / f"moss_{os.getpid()}_{i}.png"
                self.emit({"type": "status", "msg": f"Generating image {i + 1} of {n} (MOSS-Image)…"})
                started = time.time()

                def on_event(ev, _i=i):
                    if ev.get("type") == "progress":
                        self.emit({"type": "progress", "image": _i + 1, "images": n, "step": ev["step"], "steps": ev["steps"]})
                    elif ev.get("type") == "status":
                        self.emit(ev)

                self.skip_event.clear()
                try:
                    self.moss.request({"cmd": "generate", "prompt": p_i, "negative_prompt": n_i or "", "width": width,
                                       "height": height, "steps": steps, "cfg": cfg, "seed": img_seed, "out": str(out)},
                                      on_event=on_event, skip_event=self.skip_event)
                except moss_engine.CancelledError:
                    if self.skip_event.is_set() and not self.cancel_event.is_set():
                        self.skip_event.clear()
                        self.emit({"type": "skipped", "image": i + 1, "images": n})
                        continue
                    raise
                image = Image.open(out).convert("RGB")
                try:
                    out.unlink()
                except OSError:
                    pass
                secs = time.time() - started
                print(f"[timing] MOSS-Image {i + 1}/{n}: {secs:.1f}s at {width}x{height} turbo={bool(spec.get('moss_turbo'))}",
                      flush=True)
                settings = {"model": model_id, "prompt": p_i, "negative_prompt": n_i, "width": image.width,
                            "height": image.height, "steps": 15 if spec.get("moss_turbo") else steps, "cfg": cfg,
                            "sampler": "flow_euler" + (" · turbo" if spec.get("moss_turbo") else ""), "seed": img_seed,
                            "engine": "moss", "loras": [], "seconds": round(secs, 1)}
                if o_i and o_i != p_i:
                    settings["original_prompt"] = o_i
                if args.get("restore_faces"):
                    image, face_info = self._restore_faces(image, args)
                    if face_info:
                        settings["face_restore"] = face_info
                path = self._save(image, settings)
                record = self.library.add_image(path, settings, kind="generated")
                record["thumb"] = str(thumbs.make(record["id"], path, image) or "") or None
                record = self._people_check(self._content_check(record, image), image)
                results.append(record)
                self.emit({"type": "image", "record": record})
            return results
        except moss_engine.CancelledError:
            raise Cancelled()
        except moss_engine.EngineError as e:
            raise UserError(str(e))
        finally:
            if 0 < ram < 24 and self.moss is not None:
                self.moss.stop()  # give the memory back to macOS between jobs on smaller Macs

    def _mlx_load(self, spec, lora_reqs, allow_download=True, variant=None, extra_loras=None):
        """Get a fast-engine (MLX) model ready in its worker process. Returns this Mac's RAM in GB.
        The worker keeps the model loaded; asking again for the same model and LoRAs costs nothing."""
        import mlx_engine
        import net
        fam = spec["family"]
        fam_name = {"ernie": "ERNIE", "flux": "FLUX", "flux2": "FLUX.2"}.get(fam, fam)
        if self.pipe is not None or self.enhancer is not None:
            self.unload(keep_mlx=True)  # only one big model in memory at a time
            self.enhancer = None
        ram = total_ram_gb()
        if 0 < ram < 16:
            self.drop_helpers()
        if self.mlx is None:
            self.mlx = mlx_engine.MlxEngine(self.emit, self.cancel_event, self.pause_event)
        if not self.spec_cached(spec):
            size = spec["download_gb"] + (0 if mlx_engine.installed() else spec.get("engine_gb", 0))
            free = free_disk_gb(_hub_dir())
            if free < size + 3:
                raise UserError(f"{spec['label']} needs about {size:.0f} GB of downloads but only {free} GB "
                                "is free. Free up some space first.")
        # LoRAs (.safetensors) are baked in when the model loads; changing them reloads it.
        loras = []
        available = {l["name"]: l for l in lora_scan.all_loras(paths.LORAS_DIR)}
        for r in lora_reqs:
            info = available.get(r["name"])
            if info is None or info["family"] not in (fam, "unknown") or not info["file"].endswith(".safetensors"):
                raise UserError(f"LoRA '{r['name']}' can't be used with {fam_name}.")
            # FLUX.1 and FLUX.2 LoRAs look alike; a type picked by hand can be wrong: the file itself decides
            other = {"flux2": "flux", "flux": "flux2"}.get(fam)
            if other and info.get("detected_family") == other:
                raise UserError(f"LoRA '{r['name']}' was made for {'FLUX.1' if other == 'flux' else 'FLUX.2'}, so it "
                                f"can't be used with {fam_name}. If its type was set by hand, set it back in Models → "
                                "Fix a type, or pick a matching model.")
            loras.append([info["file"], float(r.get("weight", 1.0))])
        for path, scale in extra_loras or []:  # LoRAs the app itself uses (e.g. the outpaint LoRA for Expand)
            loras.append([str(path), float(scale)])
        if not mlx_engine.installed():
            if not net.is_online():
                raise UserError("offline: the fast engine isn't installed yet")  # retried online by the backend
        self.mlx.install()
        if self.cancel_event.is_set():
            raise Cancelled()
        load_started = time.time()
        self.mlx.request({"cmd": "load", "repo": spec["mlx_repo"], "base": spec.get("mlx_base"),
                          "revision": spec.get("mlx_revision"), "variant": variant,
                          "quantize": spec.get("mlx_quantize"), "patterns": spec.get("mlx_patterns"),
                          "label": spec["label"], "size_gb": spec.get("download_gb"),
                          "loras": loras, "low_ram": 0 < ram < 20,
                          "allow_download": bool(allow_download) and net.is_online()},
                         on_event=lambda ev: ev.get("type") == "status" and self.emit(ev))
        print(f"[timing] fast engine ready in {time.time() - load_started:.1f}s", flush=True)
        return ram

    # ------------------------------------------------------------ keeping a model ready
    def preload(self, args, force=False, low_memory=False):
        """Load a model (and its LoRAs) while nothing else is running, so the next Generate starts at once.
        Never downloads anything: a model that isn't on this Mac yet is skipped."""
        import mlx_engine
        model_id = args.get("model")
        spec = all_models().get(model_id)
        if spec is None:
            return {"skipped": "unknown model"}
        if not self.spec_cached(spec):
            return {"skipped": "not downloaded yet"}
        started = time.time()
        loras = [l for l in (args.get("loras") or []) if isinstance(l, dict) and l.get("name")]
        ram = total_ram_gb()
        if spec.get("moss"):  # MOSS-Image loads its parts one at a time for each picture: nothing to keep ready
            return {"skipped": "MOSS-Image loads when you make a picture"}
        try:
            if spec.get("engine") == "mlx":
                if not mlx_engine.installed():
                    return {"skipped": "the fast engine isn't installed yet"}
                if ram and ram < (spec.get("min_ram_gb") or 8) * 0.5 and not force:
                    return {"skipped": "needs more memory than this Mac has"}
                try:
                    self._mlx_load(spec, loras, allow_download=False)
                except mlx_engine.CancelledError:
                    raise Cancelled()
                except mlx_engine.EngineError as e:
                    return {"skipped": str(e)}
            else:
                if self.is_lean(spec):
                    return {"skipped": "loaded with each picture to save memory"}
                self.load(model_id, force=force, low_memory=low_memory)
                if self.cancel_event.is_set():
                    raise Cancelled()
                try:
                    self.apply_loras(loras)  # fusing LoRAs takes a few seconds too
                except UserError:
                    pass  # a LoRA that doesn't fit is reported when you press Generate
        except UserError as e:  # e.g. not enough memory: Generate reports it properly; nothing to do here
            return {"skipped": str(e)}
        secs = round(time.time() - started, 1)
        self.emit({"type": "status", "msg": f"{spec['label']} is ready"})
        self.emit({"type": "model_ready", "model": model_id})
        return {"ready": model_id, "seconds": secs}

    # ------------------------------------------------------------ content filter
    def _get_nsfw_checker(self):
        import nsfw
        if self.nsfw_checker is None:
            if not nsfw.checker_installed():
                self.emit({"type": "status", "msg": f"Downloading the picture checker (first time only, "
                                                    f"~{int(nsfw.CHECKER_GB * 1000)} MB)…"})
            self.nsfw_checker = nsfw.Checker()
        return self.nsfw_checker

    def _content_check(self, record, image):
        """Score a new picture with the content filter's checker, when that's switched on and installed.
        Never downloads, and never stops a job: a failed check just leaves the picture unchecked."""
        if not self.check_pictures or record is None:
            return record
        try:
            import nsfw
            if self.nsfw_checker is None and not nsfw.checker_installed():
                return record
            score = self._get_nsfw_checker().score(image)
            self.library.set_nsfw_score(record["id"], score)
            fresh = self.library.get_image(record["id"])
            if fresh:
                fresh["thumb"] = record.get("thumb")
                return fresh
        except Exception as e:
            print("[engine] picture check failed:", e, flush=True)
        return record

    def _face_finder_ready(self):
        """The face finder, if its files are on this Mac (it's the one Restore faces uses). Never downloads here."""
        import detailer
        if self.face_finder is None:
            d = paths.WEIGHTS_DIR / "facexlib"
            if not any(d.glob("detection_Resnet50*")):
                return None
            self.face_finder = detailer.FaceFinder(d)
        return self.face_finder

    def _people_check(self, record, image):
        """Count the faces in a new picture (for Hide People), unless its prompt already says it shows a person."""
        if record is None or record.get("people_prompt"):
            return record
        try:
            ff = self._face_finder_ready()
            if ff is None:
                return record
            small = image.copy()
            small.thumbnail((640, 640))
            n = len(ff.find(small, min_score=0.7, min_size=20))
            self.library.set_faces(record["id"], n)
            record = dict(record, faces=n)
        except Exception as e:
            print("[engine] face check failed:", e, flush=True)
        return record

    def people_scan(self, args):
        """Look for faces in every Library picture not checked yet (Hide People). Downloads the face finder once."""
        from PIL import Image
        import detailer
        todo = self.library.faces_unchecked()
        if self.face_finder is None:
            self.emit({"type": "status", "msg": "Loading the face finder (first time ~100 MB)…"})
            self.face_finder = detailer.FaceFinder(paths.WEIGHTS_DIR / "facexlib")
        done, found, missing = 0, 0, 0
        started = time.time()
        for i, (image_id, path) in enumerate(todo):
            if self.cancel_event.is_set():
                break
            if i % 5 == 0:
                self.emit({"type": "progress", "image": 1, "images": 1, "step": i, "steps": len(todo)})
                self.emit({"type": "status", "msg": f"Looking for people {i + 1} of {len(todo)}…"})
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((640, 640))
                    n = len(self.face_finder.find(im, min_score=0.7, min_size=20))
            except (OSError, ValueError):
                missing += 1
                continue
            self.library.set_faces(image_id, n)
            done += 1
            found += 1 if n else 0
        print(f"[timing] looked for faces in {done} pictures in {time.time() - started:.1f}s", flush=True)
        return {"checked": done, "with_people": found, "missing": missing}

    def nsfw_scan(self, args):
        """Check every picture in the Library that hasn't been checked yet (downloads the checker once)."""
        from PIL import Image
        import nsfw
        todo = self.library.unchecked()
        checker = self._get_nsfw_checker()
        done, flagged, missing = 0, 0, 0
        started = time.time()
        for i, (image_id, path) in enumerate(todo):
            if self.cancel_event.is_set():
                break
            if i % 5 == 0:
                self.emit({"type": "progress", "image": 1, "images": 1, "step": i, "steps": len(todo)})
                self.emit({"type": "status", "msg": f"Checking pictures {i + 1} of {len(todo)}…"})
            try:
                with Image.open(path) as im:
                    im.thumbnail((512, 512))
                    score = checker.score(im)
            except (OSError, ValueError):
                missing += 1
                continue
            self.library.set_nsfw_score(image_id, score)
            done += 1
            if score >= nsfw.THRESHOLD:
                flagged += 1
        print(f"[timing] checked {done} pictures in {time.time() - started:.1f}s", flush=True)
        return {"checked": done, "flagged": flagged, "missing": missing, "left": len(todo) - done - missing,
                "counts": self.library.nsfw_counts()}

    def _tune_for_size(self, pipe, width, height):
        """Memory-saving tricks cost speed, so only switch them on when the image is big enough to need
        them on this Mac. (Before, they were always on below 24-64 GB, which slowed small images ~15-25%.)"""
        ram = total_ram_gb() or 64
        mp = width * height / 1e6
        heavy = self.family == "sdxl" or self.family == "ernie"
        # On Apple Silicon below 64 GB, attention slicing *helps* speed: it keeps memory pressure down so
        # macOS doesn't start swapping (turning it off made 512px images on an 8 GB Mac far slower).
        slice_attn = self.family in ("sd15", "sdxl") and (ram < 64)
        # (with an image prompt adapter only the self-attention layers are sliced: the adapter lives in the
        # cross-attention layers. Leaving slicing off with one made 8 GB Macs swap: 90+ s a step)
        tile_vae = (ram < 12 and mp > 0.7) or (ram < 24 and mp > 1.1) or mp > 2.5
        want = (slice_attn, tile_vae, self.ip_key)
        if self._tuning == want:
            return
        def call(names):
            for obj in (pipe, getattr(pipe, "vae", None)):
                for n in names:
                    fn = getattr(obj, n, None) if obj is not None else None
                    if fn:
                        try:
                            fn()
                            return
                        except Exception:
                            pass
        if self.family in ("sd15", "sdxl") and not self.ip_key:
            call(["enable_attention_slicing"] if slice_attn else ["disable_attention_slicing"])
        elif self.family in ("sd15", "sdxl") and slice_attn:
            # an image prompt adapter is loaded: pipe-wide slicing would swap out its layers (and the next picture
            # fails), so only the self-attention layers (attn1, where the big tables are) are sliced
            _slice_self_attention(getattr(pipe, "unet", None))
        call(["enable_vae_tiling", "enable_tiling"] if tile_vae else ["disable_vae_tiling", "disable_tiling"])
        call(["enable_vae_slicing", "enable_slicing"])  # only matters for several images at once; no speed cost
        self._tuning = want
        print(f"[engine] attention slicing={'on' if slice_attn else 'off'}, VAE tiling={'on' if tile_vae else 'off'} "
              f"for {width}x{height}", flush=True)

    def free_original(self, model_id):
        """Delete the original (fp16) download of a model that has a saved quantized copy."""
        spec = all_models().get(model_id)
        if not spec or not self.local_copy(spec):
            raise UserError("There's no converted copy of this model yet, so the original is still needed.")
        if self.pipe is not None and self.loaded_key and self.loaded_key[0] == model_id:
            self.unload()
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        revs = [r.commit_hash for repo in cache.repos if repo.repo_id == spec["repo"] for r in repo.revisions]
        if not revs:
            return {"freed_gb": 0}
        strategy = cache.delete_revisions(*revs)
        freed = strategy.expected_freed_size
        strategy.execute()
        return {"freed_gb": round(freed / 1e9, 1)}

    def delete_model(self, args):
        """Delete a downloaded model's files from this Mac (Models tab → Delete). Files another downloaded model
        still uses are kept. It downloads again the next time it's used."""
        model_id = args.get("model")
        models = all_models()
        spec = models.get(model_id)
        if spec is None:
            raise UserError(f"Unknown model: {model_id}")
        if ":" in model_id and not model_id.startswith("hf:"):
            raise UserError("This is your own file: move it to the Trash from the Models tab instead.")
        self.unload()  # never delete what's in memory (also stops the fast engine)
        self.enhancer = None

        def needs(v):
            """Hugging Face repos a model needs on disk."""
            out = {v.get("mlx_repo"), v.get("gguf_repo")}
            if not (v.get("local_name") and self.local_copy(v)):  # a saved 8-bit copy doesn't need its original
                out.add(v.get("repo"))
            return {r for r in out if r and not v.get("local_dir")}

        mine = {r for r in (spec.get("repo"), spec.get("mlx_repo"), spec.get("gguf_repo")) if r and not spec.get("local_dir")}
        kept = []
        for k, v in models.items():
            if k == model_id or ":" in k or not self.spec_cached(v):
                continue
            shared = mine & needs(v)
            if shared:
                mine -= shared
                kept.append(v["label"])
        freed = 0
        if mine:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            revs = [r.commit_hash for repo in cache.repos if repo.repo_id in mine for r in repo.revisions]
            if revs:
                strategy = cache.delete_revisions(*revs)
                freed += strategy.expected_freed_size
                strategy.execute()
        if spec.get("local_name"):
            d = paths.QUANT_DIR / spec["local_name"]
            if d.exists():
                freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
        gb = round(freed / 1e9, 1)
        self.emit({"type": "status", "msg": f"Deleted {spec['label']} · {gb} GB freed"
                                            + (f" (kept files {kept[0]} also uses)" if kept else "")})
        return {"model": model_id, "freed_gb": gb, "kept_for": kept}

    def _warn_if_short_of_memory(self, family):
        """Before an SD 1.5 / SDXL picture on a Mac under 16 GB: if other apps have already taken the memory (little
        free, several GB already moved to disk), say so up front. That, not the model, is what makes a picture take
        30-100 s a step instead of a few (seen on an 8 GB Mac with a browser full of tabs open)."""
        ram = total_ram_gb()
        if family not in ("sd15", "sdxl") or not 0 < ram < 16:
            return
        free, swap = _free_ram_gb(), _swap_used_gb()
        need = 1.0 if family == "sd15" else 2.0
        if 0 <= free < need and swap > 1.5:
            print(f"[engine] short of memory before starting: free {free:.1f} GB, swap {swap:.1f} GB", flush=True)
            self.emit({"type": "memory_low", "free_gb": round(free, 1), "swap_gb": round(swap, 1), "family": family})

    def release_memory(self):
        """Run after every job: hand cached GPU memory back to macOS (helps 8 GB Macs a lot)."""
        gc.collect()
        try:
            t = self._torch
            if t is not None and t.backends.mps.is_available():
                t.mps.empty_cache()
        except Exception:
            pass

    def drop_helpers(self):
        """Free optional helper models (tagger, face restorer, prompt enhancer, upscaler)."""
        self.interrogator = None
        self.face_restorer = None
        self.prompt_enhancer = None
        self.enhancer = None
        self.depth_estimator = None
        self.face_finder = None
        self.face_swapper = None
        self.pose_detector = None
        self.nsfw_checker = None
        self.release_memory()

    # ------------------------------------------------------------ LoRAs
    def apply_loras(self, requested, extra=None):
        """requested: list of {name, weight}. Loads/unloads adapters as needed.
        extra: internal LoRAs, e.g. the fast-mode LCM LoRA: [(adapter_id, file, weight, label)]."""
        available = {l["name"]: l for l in lora_scan.all_loras(paths.LORAS_DIR)}
        wanted = list(extra or [])
        for r in requested or []:
            info = available.get(r["name"])
            if info is None:
                raise UserError(f"LoRA not found: {r['name']} (looked in {paths.LORAS_DIR})")
            if info["family"] not in (self.family, "unknown"):
                raise UserError(
                    f"LoRA '{r['name']}' is for {info['family'].upper()}, but the loaded model is {self.family.upper()}.")
            wanted.append((_adapter_id(r["name"]), info["file"], float(r.get("weight", 1.0)), r["name"]))

        wanted_ids = {w[0] for w in wanted}
        stale = [a for a in self.adapters if a not in wanted_ids]
        fuse_key = tuple((w[0], round(w[2], 4)) for w in wanted)
        if self.fused_key is not None:
            if fuse_key == self.fused_key:
                return  # same LoRAs and weights: already baked into the model
            try:
                self.pipe.unfuse_lora()
            except Exception as e:
                print("[engine] unfuse failed, reloading model next time:", e, flush=True)
                self.loaded_key = None
            self.fused_key = None
        if not wanted:
            if self.adapters:
                self.pipe.unload_lora_weights()
                self.adapters = {}
            return
        if stale:
            self.pipe.delete_adapters(stale)
            for a in stale:
                self.adapters.pop(a, None)
        for aid, file, _, name in wanted:
            if aid not in self.adapters:
                self.emit({"type": "status", "msg": f"Loading LoRA {name}…"})
                try:
                    if file.endswith(".tdict"):  # DiffusionBee LoRA: convert in memory
                        import tdict_reader
                        self.pipe.load_lora_weights(tdict_reader.TDict(file).lora_state_dict(), adapter_name=aid)
                    else:
                        self.pipe.load_lora_weights(file, adapter_name=aid)
                except Exception as e:
                    raise UserError(f"Couldn't load LoRA '{name}': {e}")
                self.adapters[aid] = file
        self.pipe.set_adapters([w[0] for w in wanted], adapter_weights=[w[2] for w in wanted])
        # Speed: bake the LoRAs into the model weights ("fuse") so each step runs at plain-model speed
        # instead of computing every LoRA on the side. Undone automatically when the mix changes.
        # Not possible on 8-bit/4-bit (quantized) weights; those keep the slower live LoRAs.
        if self.fuse_ok:
            t0 = time.time()
            try:
                self.pipe.fuse_lora(adapter_names=[w[0] for w in wanted], lora_scale=1.0)
                self.fused_key = fuse_key
                print(f"[timing] fused {len(wanted)} LoRA(s) in {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                print("[engine] couldn't fuse LoRAs (using them live instead):", e, flush=True)
                try:
                    self.pipe.unfuse_lora()
                except Exception:
                    pass

    def apply_embeddings(self, prompt, negative):
        """Embeddings (textual inversions, e.g. EasyNegative): if a prompt mentions one made for this model's
        type (SD 1.5 or SDXL) by name, load it. Returns the prompts with the name spelled exactly as the token."""
        if self.family not in ("sd15", "sdxl") or not hasattr(self.pipe, "load_textual_inversion"):
            return prompt, negative
        try:
            import embeddings as emb_scan
            found = emb_scan.for_family(self.family)
        except Exception:
            return prompt, negative
        for e in found:
            tok = e["name"]
            pat = re.compile(r"(?<![\w-])" + re.escape(tok) + r"(?![\w-])", re.IGNORECASE)
            if not (pat.search(prompt) or pat.search(negative)):
                continue
            prompt, negative = pat.sub(lambda _m: tok, prompt), pat.sub(lambda _m: tok, negative)
            if tok in self.embeddings:
                continue
            self.emit({"type": "status", "msg": f"Loading embedding {tok}…"})
            try:
                vec = emb_scan.load_vectors(e)
                pipe = self.pipe
                if self.family == "sdxl":
                    pipe.load_textual_inversion({"emb_params": vec["clip_l"]}, token=tok,
                                                text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                    pipe.load_textual_inversion({"emb_params": vec["clip_g"]}, token=tok,
                                                text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
                else:
                    pipe.load_textual_inversion({"emb_params": vec["clip_l"]}, token=tok)
                self.embeddings.add(tok)
            except Exception as err:
                print(f"[engine] couldn't load embedding {tok}: {err}", flush=True)
                self.emit({"type": "status", "msg": f"Couldn't load the embedding {tok} ({type(err).__name__}); "
                                                    "it's used as a plain word"})
        return prompt, negative

    # ------------------------------------------------------------ generation
    def get_prompt_enhancer(self):
        if self.prompt_enhancer is None:
            from postprocess.prompt_enhance import PromptEnhancer, DOWNLOAD_GB
            pe = PromptEnhancer(self.device(), emit=self.emit, ram_gb=total_ram_gb() or 8)
            if not self._is_cached(pe.repo):
                gb = DOWNLOAD_GB[pe.repo]
                free = free_disk_gb(_hub_dir())
                if free < gb + 2:
                    raise UserError(f"The prompt enhancer is a ~{gb} GB download but only {free} GB is free.")
                self.emit({"type": "status", "msg": f"Downloading the prompt enhancer (first time only, ~{gb:g} GB)…"})
            self.prompt_enhancer = pe
        return self.prompt_enhancer

    def enhance_prompt(self, args):
        """Rewrite a short prompt into a detailed one. Returns {"prompt": ...}."""
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            raise UserError("Type a prompt first.")
        family = args.get("family")
        if not family and args.get("model"):
            family = (all_models().get(args["model"]) or {}).get("family")
        family = family or "sdxl"
        ram = total_ram_gb()
        if 0 < ram < 12 and self.family in ("sdxl", "ernie") and self.pipe is not None:
            self.unload()  # 8 GB Macs: a big image model and the enhancer don't fit together
        text = self.get_prompt_enhancer().enhance(prompt, int(args.get("width") or 1024),
                                                  int(args.get("height") or 1024),
                                                  cancel_event=self.cancel_event, family=family,
                                                  contrast=bool(args.get("contrast")))
        if self.cancel_event.is_set():
            raise Cancelled()
        from postprocess.prompt_enhance import style_for
        return {"prompt": text, "original": prompt, "style": style_for(family, bool(args.get("contrast")))}

    def plan_regions(self, args):
        """✨ Plan the layout (Create → Regions): one sentence -> areas (layout, ratios, prompts) and the scene prompt.
        The prompt enhancer's small model writes one line per area; if its answer can't be used, the sentence's own
        position words are read (region_plan.from_words)."""
        import region_plan
        sentence = (args.get("prompt") or "").strip()
        if not sentence:
            raise UserError("Describe the picture first, e.g. “a knight on the left fighting a dragon on the right "
                            "under a stormy sky”.")
        if args.get("words_only") or not region_plan.needs_model(sentence):
            try:  # you said where things go: your places are used as they are
                return region_plan.plan(sentence)
            except ValueError as e:
                raise UserError(str(e))
        ram = total_ram_gb()
        if 0 < ram < 12 and self.pipe is not None and self.family in ("sdxl", "ernie", "flux", "flux2", "anima"):
            self.unload()  # 8 GB Macs: a big image model and a language model don't fit together
        import vlm
        planners = []
        if vlm.ready():  # Qwen3.5 4B (the picture reader, if it's set up): far better plans than the 0.5B enhancer
            def big():
                q = region_plan.SYSTEM + "\n\n" + "\n\n".join(f"Description: {a}\n{b}" for a, b in region_plan.EXAMPLES)
                self.emit({"type": "status", "msg": f"Planning the layout with {vlm.LABEL}…"})
                return vlm.chat(q + f"\n\nDescription: {sentence}\n", emit=self.emit, cancel_event=self.cancel_event)
            planners.append(("Qwen3.5 4B", big))
            planners.append(("Qwen3.5 4B", lambda: vlm.chat(  # once more, if it kept the subjects together
                region_plan.SYSTEM + f"\n\nDescription: {sentence}\nGive every subject its own place line (at least "
                "two place lines).\n", emit=self.emit, cancel_event=self.cancel_event)))
        planners.append(("prompt enhancer", lambda: self.get_prompt_enhancer().plan_layout(sentence, cancel_event=self.cancel_event)))
        last = None
        for name, ask in planners:
            try:
                raw = ask()
            except Exception as err:  # not set up / offline before its first download: try the next one
                if self.cancel_event.is_set():
                    raise Cancelled()
                print(f"[regions] {name} unavailable:", err, flush=True)
                continue
            if self.cancel_event.is_set():
                raise Cancelled()
            print(f"[regions] {name} plan:", raw.replace("\n", " | ")[:400], flush=True)
            try:
                return dict(region_plan.plan(sentence, raw), planner=name)
            except ValueError as e:
                last = e
        raise UserError(str(last) if last else "Couldn't plan the layout: the planner models aren't available.")

    def _ernie_embeddings(self, spec, texts):
        """Lean mode: encode prompts with a temporarily loaded text encoder, then free it."""
        repo = spec["repo"]
        missing = [x for x in texts if (repo, x) not in self.embed_cache]
        if missing:
            import diffusers
            t = self.torch()
            if total_ram_gb() < 12 and self.pipe is not None:
                self.unload()  # 8 GB Macs: don't hold the image model while the text encoder is in memory
            self.emit({"type": "status", "msg": "Encoding prompt…"})
            te = diffusers.ErnieImagePipeline.from_pretrained(
                repo, transformer=None, vae=None, pe=None, pe_tokenizer=None, torch_dtype=t.bfloat16).to(self.device())
            for x in missing:
                self.embed_cache[(repo, x)] = te.encode_prompt(x, te._execution_device)[0].detach()
            del te
            gc.collect()
            if t.backends.mps.is_available():
                t.mps.empty_cache()
            if len(self.embed_cache) > 32:  # keep the cache small
                for k in list(self.embed_cache)[:-32]:
                    del self.embed_cache[k]
        return [self.embed_cache[(repo, x)] for x in texts]

    def generate(self, args, force=False, low_memory=False, keep=None):
        """Make images. Modes: text to image (Create), image to image, ControlNet, inpaint (mask), outpaint
        (expand the canvas). Extras: Hi-res fix, face detailer, image prompt (IP-Adapter), wildcards,
        long/weighted prompts, CLIP skip.
        keep: optional list; the PIL images are appended to it (used by the comparison grid)."""
        from PIL import Image
        import wildcards
        import prompt_edit
        import prompt_weights
        model_id = args["model"]
        models = all_models()
        if model_id not in models:
            raise UserError(f"Unknown model: {model_id}")
        spec = models[model_id]
        d = spec["defaults"]
        family = spec["family"]
        is_sd = family in ("sd15", "sdxl") and spec.get("engine") != "mlx"
        rg0 = args.get("regions") if isinstance(args.get("regions"), dict) else None
        if (rg0 and rg0.get("on", True) and spec.get("engine") == "mlx" and family in ("flux", "ernie", "flux2") and not args.get("_raw")
                and not args.get("init_image") and len([x for x in rg0.get("prompts") or [] if isinstance(x, str) and x.strip()]) >= 2):
            return self.regions_by_parts(args, force=force)
        if spec.get("inpaint") and not (args.get("mask") or isinstance(args.get("outpaint"), dict)):
            raise UserError(f"{spec['label']} is an inpainting model: it only fills in painted areas. It's used by "
                            "Image to image → Inpaint and Expand; pick another model to make or change whole pictures.")

        width = int(args.get("width") or d["width"])
        height = int(args.get("height") or d["height"])
        steps = int(args.get("steps") or d["steps"])
        cfg = float(args.get("cfg") if args.get("cfg") not in (None, "") else d["cfg"])
        n = max(1, min(int(args.get("num_images") or 1), 64))  # the fader goes to 16; typed amounts up to 64
        seed = args.get("seed")
        try:
            seed = int(seed) if seed not in (None, "") else -1
        except (TypeError, ValueError):
            raise UserError("The seed must be a whole number (or -1 for random).")
        seed = random.randint(0, 2**32 - 1) if seed < 0 else seed % 2**32
        prompt_t = args.get("prompt") or ""
        negative_t = args.get("negative_prompt") or ""
        if not prompt_t.strip():
            raise UserError("Type a prompt first.")

        def open_image(path, what):
            import pictures
            try:
                path = pictures.prepare(path)  # HEIC, TIFF, RAW… are converted by macOS first
                return Image.open(path).convert("RGB")
            except pictures.PictureError as e:
                raise UserError(f"Couldn't open the {what}: {e}")
            except Exception:
                raise UserError(f"Couldn't open the {what}.")

        # image-to-image
        init_image, strength = None, None
        if args.get("init_image"):
            if not is_sd and family != "anima" and not (spec.get("engine") == "mlx" and family in ("flux", "ernie")):
                raise UserError("Image-to-image works with SD 1.5, SDXL, Anima, FLUX.1 and ERNIE (fast engine). "
                                "For FLUX.2 Klein use ✨ Edit with words.")
            init_image = open_image(args["init_image"], "start image for image-to-image")
            strength = max(0.05, min(1.0, float(args.get("strength", 0.6))))

        # outpaint: a bigger canvas with the picture inside; the new area gets filled in
        mask = None
        outpaint = args.get("outpaint") if isinstance(args.get("outpaint"), dict) else None
        if outpaint and init_image is not None:
            import detailer
            pads = {k: max(0, int(outpaint.get(k) or 0)) for k in ("left", "right", "top", "bottom")}
            if not any(pads.values()):
                raise UserError("Choose how far to expand the picture (at least one side).")
            init_image, mask = detailer.outpaint_canvas(init_image, **pads)
            strength = 1.0  # the new area starts from scratch
        elif args.get("mask") and init_image is not None:
            import detailer
            try:
                mask = detailer.decode_mask(args["mask"], init_image.size)
            except Exception:
                raise UserError("Couldn't read the painted mask.")
            if not mask.getbbox():
                raise UserError("Paint over the part you want to change first.")
            mask = detailer.soften(mask, float(args.get("mask_blur", 4)))
        if init_image is not None and (args.get("use_init_size") or outpaint):
            scale = min(1.0, _max_side(family) / max(init_image.size))
            width, height = int(init_image.width * scale), int(init_image.height * scale)

        # ControlNet (Image to image tab): follow a control map made from an image
        control = args.get("control") if isinstance(args.get("control"), dict) and (args.get("control") or {}).get("type") else None
        control_src = None
        cnet_union, cnet_mode = False, None
        if control:
            if mask is not None:
                raise UserError("Inpainting/expanding and ControlNet can't be combined yet: switch one off.")
            import controlnet as cnet
            if not cnet.is_known(control["type"]):
                raise UserError(f"Unknown ControlNet type: {control['type']}")
            cn_fam = cnet.family_of(control["type"])
            if not is_sd or family != cn_fam:
                names = {"sd15": "an SD 1.5", "sdxl": "an SDXL (or Pony, Illustrious, NoobAI)"}
                raise UserError(f"This ControlNet is made for {names.get(cn_fam, 'another type of')} model, so it can't be "
                                f"used with {spec.get('label') or model_id}. Pick {names.get(cn_fam, 'a matching')} model, "
                                "or a ControlNet made for this one.")
            src_path = args.get("control_image") or args.get("init_image")
            if not src_path:
                raise UserError("Add an image for the ControlNet to follow.")
            control_src = open_image(src_path, "ControlNet image")
            cnet_union = cnet.is_union(control["type"])
            cnet_mode = cnet.union_mode(control["type"]) if cnet_union else None
            if args.get("use_init_size") and init_image is None:
                scale = min(1.0, _max_side(family) / max(control_src.size))
                width, height = int(control_src.width * scale), int(control_src.height * scale)

        # CLIP skip (A1111 numbering: 1 = off, 2 = skip the last text-encoder layer, ...). SD 1.5 and SDXL only.
        try:
            clip_skip = int(args.get("clip_skip") or 1)
        except (TypeError, ValueError):
            clip_skip = 1
        clip_skip = max(1, min(12, clip_skip)) if is_sd else 1
        if family == "sdxl":
            clip_skip = max(2, clip_skip)  # SDXL always reads the second-to-last layer: that *is* "CLIP skip 2"
        extra_skip = clip_skip - (2 if family == "sdxl" else 1)  # diffusers' clip_skip
        mult = 16 if family in ("ernie", "flux", "flux2", "anima") else 8
        width, height = max(mult * 8, width - width % mult), max(mult * 8, height - height % mult)

        # extras
        hires = args.get("hires") if isinstance(args.get("hires"), dict) and args["hires"].get("on") else None
        if hires and (not is_sd or init_image is not None or control):
            hires = None  # Hi-res fix is for text to image with SD 1.5 / SDXL
        det = args.get("face_detailer") if isinstance(args.get("face_detailer"), dict) and args["face_detailer"].get("on") else None
        if det and not is_sd:
            det = None
        ip = args.get("ip_adapter") if isinstance(args.get("ip_adapter"), dict) and args["ip_adapter"].get("image") else None
        ip_image = None
        if ip:
            if not is_sd:
                raise UserError("Image prompts (IP-Adapter) work with SD 1.5 and SDXL models.")
            ip_image = open_image(ip["image"], "image prompt picture")

        # per-image prompts: wildcards ({a|b}, __file__) pick from each image's seed
        seeds = [(seed + i) % 2**32 for i in range(n)]
        # 🎲 Small change (variation seed, SD 1.5 / SDXL): -1 picks one; strength 0 leaves the picture as it was
        var_strength = max(0.0, min(1.0, float(args.get("var_strength") or 0)))
        var_seed = int(args.get("var_seed") if args.get("var_seed") not in (None, "", -1) else random.randrange(2**32)) % 2**32
        var_on = bool(is_sd and var_strength > 0)
        prompts = [wildcards.for_image(prompt_t, s) for s in seeds]
        negatives = [wildcards.for_image(negative_t, s) for s in seeds]
        # Pony / Illustrious / NoobAI: their quality tags (Create → "add quality tags", on by default)
        flavor = model_flavor(spec, model_id)
        if flavor and args.get("quality_tags", True):
            qt = QUALITY_TAGS[flavor]
            prompts = [add_quality_tags(x, qt["prompt"]) for x in prompts]
            negatives = [add_quality_tags(x, qt["negative"]) for x in negatives]
        originals = list(prompts)
        if args.get("enhance_prompt"):  # before the image model loads, to keep memory low
            done = {}
            for i, p in enumerate(prompts):
                if p not in done:
                    done[p] = self.enhance_prompt({"prompt": p, "width": width, "height": height,
                                                   "family": family})["prompt"]
                    self.emit({"type": "enhanced_prompt", "prompt": done[p], "original": p})
                prompts[i] = done[p]
        if self.cancel_event.is_set():
            raise Cancelled()

        if spec.get("moss"):
            return self._generate_moss(args, model_id, spec, prompts, negatives, width, height, steps, cfg, n, seed,
                                       originals, force=force)
        if spec.get("engine") == "mlx":
            return self._generate_mlx(args, model_id, spec, prompts, negatives, width, height, steps, cfg, n, seed,
                                      originals, force=force)
        if family == "anima":
            return self._generate_anima(args, model_id, spec, prompts, negatives, originals, width, height, steps, cfg,
                                        seeds, init_image, strength, force=force)

        lean = self.is_lean(spec)
        lean_enc = {}
        if lean:
            texts = sorted(set(prompts) | (set(negatives) if cfg > 1 else set()))
            for text, emb in zip(texts, self._ernie_embeddings(spec, texts)):
                lean_enc[text] = emb

        self.load(model_id, force=force, low_memory=low_memory)
        if is_sd:  # 🧱 Seamless: every edge wraps around, so the picture tiles (textures, patterns, wallpapers)
            _set_seamless(self.pipe, bool(args.get("seamless")))
        if self.cancel_event.is_set():  # stopped while the model was loading
            raise Cancelled()
        vae_used, vae_why = self._pick_vae(args.get("vae"), spec, family) if is_sd else (None, "")
        self._apply_vae(vae_used, family)
        fast = bool(args.get("fast")) and family in FAST_LORAS and mask is None
        extra = [(FAST_ADAPTER, self._fast_lora_file(family), 1.0, "Fast mode (LCM)")] if fast else None
        self.apply_loras(args.get("loras") or [], extra=extra)
        if is_sd:
            self._set_ip_adapter(ip.get("kind", "style") if ip else None)
        for i in range(n):
            prompts[i], negatives[i] = self.apply_embeddings(prompts[i], negatives[i])
        sampler_req = "lcm" if fast else args.get("sampler")
        scheduler, sampler_id = samplers.build_scheduler(self.family, sampler_req, self.base_scheduler_config)

        pipe = self.pipe
        control_map = None
        if control:
            pipe, control_map = self._controlnet_pipe(control, control_src, init_image is not None, width, height)
            if init_image is not None:
                init_image = init_image.resize((width, height))
        elif mask is not None:
            pipe = self._inpaint_pipe()
            init_image = init_image.resize((width, height), Image.LANCZOS)
            mask = mask.resize((width, height), Image.LANCZOS)
        elif init_image is not None:
            pipe = self._i2i_pipe()
            init_image = init_image.resize((width, height))
        if scheduler is not None:
            pipe.scheduler = scheduler
        self._tune_for_size(pipe, width, height)
        self._warn_if_short_of_memory(family)
        if ip:
            try:
                self.pipe.set_ip_adapter_scale(float(ip.get("scale", 0.6)))
            except Exception as e:
                print("[engine] couldn't set image prompt strength:", e, flush=True)

        import prompt_weights
        emb_cache = {}

        def text_args(p, ng):
            """prompt/negative as the pipeline wants them: plain text, or embeddings for long/weighted prompts."""
            if is_sd and prompt_weights.needs_custom(p, ng, self.pipe.tokenizer):
                key = (p, ng, clip_skip)
                if key not in emb_cache:
                    emb_cache.clear()  # keep only one set (they hold memory)
                    emb_cache[key] = prompt_weights.encode(self.pipe, family, p, ng, clip_skip, cfg)
                return dict(emb_cache[key])
            out = {"prompt": p, "negative_prompt": ng}
            if extra_skip > 0:
                _clip_skip_compat(self.pipe)
                out["clip_skip"] = extra_skip  # diffusers counts the layers skipped (A1111's 2 = diffusers' 1)
            return out

        t = self.torch()
        # 🗺 Regional prompts: each area of the picture follows its own prompt (SD 1.5 / SDXL)
        reg_proc = None
        rg = args.get("regions") if isinstance(args.get("regions"), dict) else None
        rg_prompts = [p_.strip() for p_ in (rg or {}).get("prompts") or [] if isinstance(p_, str) and p_.strip()]
        if rg and len(rg_prompts) >= 2:
            if not is_sd:
                raise UserError("Regional prompts work with SD 1.5 and SDXL models.")
            if ip_image is not None:
                raise UserError("Regional prompts and an image prompt (IP-Adapter) can't be combined yet: switch one off.")
            import regions
            boxes = regions.parse_layout(rg.get("layout") or "cols", rg.get("ratios") or "", len(rg_prompts))
            # each area follows its own prompt; the main prompt reaches it through the "keep one scene" blend
            # (adding the whole main prompt to every area drowned the areas' subjects in the scene)
            common = (prompts[0] if rg.get("add_main") else "").strip()
            embeds = []
            for rp in rg_prompts:
                full = f"{common}, {rp}" if common else rp
                with t.no_grad():
                    if family == "sdxl":
                        embeds.append(self.pipe.encode_prompt(full, device=self.device(), num_images_per_prompt=1,
                                                              do_classifier_free_guidance=False)[0])
                    else:
                        embeds.append(self.pipe.encode_prompt(full, self.device(), 1, False)[0])
            reg_masks = regions.masks(boxes, height // 8, width // 8, self.device(), self.pipe.unet.dtype)
            reg_proc = regions.RegionalAttn(embeds, reg_masks, (height // 8, width // 8),
                                            base_weight=float(rg.get("base", 0.2)), cfg=cfg > 1)
            self.emit({"type": "status", "msg": f"Regional prompts: {len(rg_prompts)} areas"})
        results = []
        for i in range(n):
            self.wait_if_paused()
            if self.cancel_event.is_set():  # stop between images, not just between steps
                raise Cancelled()
            if i and (hires or det):
                self._tune_for_size(pipe, width, height)  # the extra passes ran at another size
            img_seed = seeds[i]
            prompt, negative = prompts[i], negatives[i]
            gen = t.Generator("cpu").manual_seed(img_seed)
            last_preview = [0.0]
            phase = {"name": ""}
            first_step, slow_warned = [0.0], [False]

            def on_step(p, step, timestep, kw, _i=i):
                self.wait_if_paused()
                if self.cancel_event.is_set():
                    raise Cancelled()
                if self.skip_event.is_set():
                    raise Skipped()
                total = getattr(p, "_num_timesteps", None) or steps
                self.emit({"type": "progress", "image": _i + 1, "images": n, "step": step + 1, "steps": total,
                           **({"phase": phase["name"]} if phase["name"] else {})})
                if self.previews and step + 1 < total and time.time() - last_preview[0] > 1.5:
                    self._emit_preview(kw.get("latents"), _i + 1)
                    last_preview[0] = time.time()
                if step == 0 or (step + 1) % 5 == 0 or step + 1 == total:
                    print(f"[timing] {phase['name'] or 'step'} {step + 1}/{total}: {time.time() - started:.1f}s elapsed",
                          flush=True)
                    if step == 0:
                        first_step[0] = time.time()
                    elif not slow_warned[0] and is_sd and step >= 4 and first_step[0]:
                        per = (time.time() - first_step[0]) / step
                        if per > 12:  # SD 1.5 / SDXL normally take 1-6 s a step on Apple Silicon
                            slow_warned[0] = True
                            swap = _swap_used_gb()
                            print(f"[engine] very slow: {per:.0f}s/step, swap in use {swap:.1f} GB, free RAM {_free_ram_gb():.1f} GB",
                                  flush=True)
                            self.emit({"type": "status", "msg":
                                       f"Very slow ({per:.0f} s a step): this Mac is short of memory and is using the disk instead. "
                                       "Close other big apps, use a smaller size or fewer LoRAs/extras, or turn on Low-memory mode "
                                       "in Settings. Tools → Force stop restarts the engine and frees its memory."})
                # [cat:dog:0.5] prompt editing: switch to the next prompt when its share of the steps is reached
                if edits and "prompt_embeds" in kw:
                    f = (step + 1) / max(1, total)
                    while edits and f >= edits[0][0]:
                        _f, emb = edits.pop(0)
                        both = p.do_classifier_free_guidance if hasattr(p, "do_classifier_free_guidance") else cfg > 1
                        pe_t = emb["prompt_embeds"].to(kw["prompt_embeds"].device, kw["prompt_embeds"].dtype)
                        kw["prompt_embeds"] = t.cat([emb["negative_prompt_embeds"].to(pe_t), pe_t]) if both else pe_t
                        if "add_text_embeds" in kw and "pooled_prompt_embeds" in emb:
                            pp = emb["pooled_prompt_embeds"].to(kw["add_text_embeds"])
                            kw["add_text_embeds"] = t.cat([emb["negative_pooled_prompt_embeds"].to(pp), pp]) if both else pp
                        phase["edit"] = phase.get("edit", 0) + 1
                        print(f"[engine] prompt edit {phase['edit']} at step {step + 1}/{total}", flush=True)
                return kw

            edits = []
            sched = prompt_edit.schedule(prompt, steps) if is_sd and prompt_edit.has_edits(prompt) else []
            call = dict(num_inference_steps=steps, guidance_scale=cfg, generator=gen, callback_on_step_end=on_step)
            if len(sched) > 1:
                # each version of the prompt is encoded up front; the pipeline starts with the first
                encs = [prompt_weights.encode(self.pipe, family, txt, negative, clip_skip, cfg) for _f, txt in sched]
                call.update(encs[0])
                edits = [(f, e) for (f, _txt), e in zip(sched[1:], encs[1:])]
                names = list(getattr(pipe, "_callback_tensor_inputs", []))
                want = [k for k in ("latents", "prompt_embeds", "add_text_embeds") if k in names]
                call["callback_on_step_end_tensor_inputs"] = want
                if "prompt_embeds" not in want:
                    edits = []
                    self.emit({"type": "status", "msg": "Prompt editing ([a:b:0.5]) isn't available in this mode: using the first prompt."})
            elif is_sd:
                call.update(text_args(sched[0][1] if sched else prompt, negative))
            if var_on and init_image is None and mask is None:
                # 🎲 Small change: start from this seed's noise nudged towards the variation seed's
                ch = int(pipe.unet.config.in_channels)
                call["latents"] = variation_noise(img_seed, (var_seed + i) % 2**32, var_strength,
                                                  (1, ch, height // 8, width // 8), pipe.unet.dtype).to(self.device())
            if ip_image is not None:
                call["ip_adapter_image"] = ip_image
            if control_map is not None:
                call.update(controlnet_conditioning_scale=float(control.get("scale", 1.0)),
                            control_guidance_start=float(control.get("start", 0.0)),
                            control_guidance_end=float(control.get("end", 1.0)))
                if cnet_union:  # Union ProMax: the map goes in as a list with its control mode
                    call.update(control_image=[control_map], control_mode=[cnet_mode])
                    if init_image is not None:
                        call.update(image=init_image, strength=strength)
                    else:
                        call.update(width=width, height=height)
                elif init_image is not None:
                    call.update(image=init_image, control_image=control_map, strength=strength)
                else:
                    call.update(image=control_map, width=width, height=height)
            elif mask is not None:
                call.update(image=init_image, mask_image=mask, strength=strength, width=width, height=height)
                if args.get("only_masked") and not outpaint:
                    call["padding_mask_crop"] = int(args.get("mask_padding", 32))
            elif init_image is not None:
                call.update(image=init_image, strength=strength)
            elif family == "ernie" and lean:
                call.update(prompt=None, prompt_embeds=[lean_enc[prompt]], width=width, height=height, use_pe=False)
                if cfg > 1:
                    call["negative_prompt_embeds"] = [lean_enc[negative]]
            else:
                call.update(width=width, height=height)
                if not is_sd:
                    call.update(prompt=prompt, negative_prompt=negative)
                if family == "ernie":
                    call["use_pe"] = False
            if family == "other":
                # newer pipelines differ in which options they take: pass only what this one accepts
                import inspect
                params = inspect.signature(pipe.__call__).parameters
                if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                    call = {k: v for k, v in call.items() if k in params}
                if "callback_on_step_end" not in call and "callback_on_step_end" in params:
                    call["callback_on_step_end"] = on_step
            self.emit({"type": "status", "msg": f"Generating image {i + 1} of {n}…"})
            started = time.time()
            self.skip_event.clear()
            extras_info = {}
            # the prompt's cross-attention keys/values are worked out once per picture, not at every step
            # (DiffusionBee's trick; same picture, a few % faster). Left off with image prompts.
            kv_on = (is_sd and not self.ip_key and getattr(pipe, "unet", None) is not None
                     and not os.environ.get("PROTOART_NO_KV_CACHE") and kv_cache.install(pipe.unet) > 0)
            reg_orig = regions.install(pipe.unet, reg_proc) if reg_proc is not None else None
            try:
                image = pipe(**call).images[0]
                if _looks_blank(image) and control:
                    import controlnet as cnet
                    if cnet.LAST_SOURCE.get(control["type"]) == "diffusionbee":
                        # the DiffusionBee copy of this ControlNet didn't work: switch to the official model
                        print(f"[engine] blank result with DiffusionBee ControlNet {control['type']}; "
                              "switching to the official model", flush=True)
                        cnet.SKIP_DBEE.add(control["type"])
                        self.controlnets.pop(control["type"], None)
                        self.cn_pipes = {k: v for k, v in self.cn_pipes.items() if k[0] != control["type"]}
                        self.emit({"type": "status", "msg": "That ControlNet file didn't work; trying the official one…"})
                        pipe, control_map = self._controlnet_pipe(control, control_src, init_image is not None,
                                                                  width, height)
                        if scheduler is not None:
                            pipe.scheduler = scheduler
                        call["generator"] = t.Generator("cpu").manual_seed(img_seed)
                        if init_image is not None:
                            call["control_image"] = control_map
                        else:
                            call["image"] = control_map
                        image = pipe(**call).images[0]
                if _looks_blank(image):
                    # a flat grey/black result means the maths broke down (usually half-precision overflow)
                    print("[engine] result came out blank; control:", control, "family:", family, flush=True)
                    raise UserError("The result came out as a flat grey image, so it wasn't saved. "
                                    + ("Try a lower ControlNet strength, a different ControlNet type, or no ControlNet. "
                                       if control else "")
                                    + "Details are in Settings → Backend log.")
                if mask is not None and not outpaint and not args.get("only_masked"):
                    # keep everything outside the mask exactly as it was (the model slightly changes it), with the
                    # painted part's light and colour matched to the original around it (no lighter patch)
                    try:
                        import color_match
                        image = color_match.match_to_original(image, init_image, mask)
                    except Exception as e:
                        print("[engine] colour match skipped:", e, flush=True)
                    image = Image.composite(image, init_image, mask)
                if hires:
                    phase["name"] = "Hi-res fix"
                    image, extras_info["hires"] = self._hires_fix(image, hires, text_args(prompt, negative), ip_image,
                                                                  img_seed, scheduler, steps, cfg, on_step)
                    phase["name"] = ""
                if reg_orig is not None:  # the face detailer redraws crops: no regions there
                    regions.remove(pipe.unet, reg_orig)
                    reg_orig = None
                if det:
                    phase["name"] = "Face detailer"
                    image, extras_info["face_detailer"] = self._face_detail(
                        image, det, prompt, negative, text_args, ip_image, img_seed, scheduler, steps, cfg, on_step)
                    phase["name"] = ""
            except Skipped:
                self.skip_event.clear()
                self.emit({"type": "status", "msg": f"Skipped image {i + 1} of {n}"})
                self.emit({"type": "skipped", "image": i + 1, "images": n})
                continue
            finally:
                if reg_orig is not None:
                    regions.remove(pipe.unet, reg_orig)
                if kv_on:
                    kv_cache.clear(pipe.unet)
            print(f"[timing] image {i + 1}/{n}: {time.time() - started:.1f}s for {steps} steps "
                  f"at {width}x{height} ({(time.time() - started) / max(1, steps):.2f}s/step)", flush=True)
            face_info = None
            if args.get("restore_faces"):
                image, face_info = self._restore_faces(image, args)
            if keep is not None:
                keep.append(image)

            settings = {
                "model": model_id, "prompt": prompt, "negative_prompt": negative,
                "width": image.width, "height": image.height, "steps": steps, "cfg": cfg,
                "sampler": sampler_id, "seed": img_seed, **({"fast": True} if fast else {}),
                **({"seamless": True} if args.get("seamless") and is_sd else {}),
                **({"regions": {k: rg.get(k) for k in ("layout", "ratios", "base")} | {"prompts": rg_prompts}}
                   if reg_proc is not None else {}),
                **({"var_seed": (var_seed + i) % 2**32, "var_strength": var_strength}
                   if var_on and init_image is None and mask is None else {}),
                **({"clip_skip": clip_skip} if is_sd else {}),
                **({"vae": vae_used} if is_sd and vae_used else {}),
                **({"vae_auto": True} if is_sd and not args.get("vae") else {}),
                "loras": [{"name": l["name"], "weight": float(l.get("weight", 1.0))} for l in args.get("loras") or []],
                "seconds": round(time.time() - started, 1),
            }
            if originals[i] != prompt:
                settings["original_prompt"] = originals[i]  # before enhancing / the wildcard template
            if args.get("enhance_prompt"):
                settings["enhance_prompt"] = True
            if prompt_t != originals[i]:
                settings["prompt_template"] = prompt_t
            if init_image is not None:
                settings.update(init_image=str(args["init_image"]), strength=strength)
            if mask is not None:
                settings["inpaint"] = ({"outpaint": {k: outpaint.get(k, 0) for k in ("left", "right", "top", "bottom")}}
                                       if outpaint else {"only_masked": bool(args.get("only_masked")),
                                                         "mask_blur": args.get("mask_blur", 4)})
            if control:
                settings["control"] = {k: control.get(k) for k in ("type", "scale", "start", "end", "already_map")}
                settings["control_image"] = str(args.get("control_image") or args.get("init_image"))
            if ip:
                settings["ip_adapter"] = {"image": str(ip["image"]), "scale": float(ip.get("scale", 0.6)),
                                          "kind": ip.get("kind", "style")}
            settings.update(extras_info)
            if face_info:
                settings["face_restore"] = face_info
            if args.get("grid_cell"):
                settings["grid_cell"] = args["grid_cell"]
            path = self._save(image, settings)
            kind = ("outpaint" if outpaint else "inpaint") if mask is not None else \
                ("img2img" if (init_image is not None or control) else "generated")
            record = self.library.add_image(path, settings, kind=kind)
            record["thumb"] = str(thumbs.make(record["id"], path, image) or "") or None
            record = self._people_check(self._content_check(record, image), image)
            results.append(record)
            self.emit({"type": "image", "record": record})
        return results

    # ------------------------------------------------------------ pipelines that share the loaded model
    @staticmethod
    def _quiet(pipe):
        try:
            pipe.set_progress_bar_config(disable=True)  # progress goes to the app, not to the log
        except Exception:
            pass

    def _pick_vae(self, choice, spec, family):
        """Create → VAE: a VAE's name, vaes.OWN (the model's own), or empty = Automatic (the right one for this
        model, see vaes.auto_for). Returns (name or None, why)."""
        import vaes
        if choice == vaes.OWN:
            return None, "the model's own"
        if choice:
            return choice, "your pick"
        name, why = vaes.auto_for(spec.get("file"), family)
        if name:
            print(f"[vae] automatic: {name} ({why})", flush=True)
        return name, why

    def _apply_vae(self, name, family):
        """Use a VAE file instead of the model's own (Create → VAE). The model's own is kept and put back when
        you pick "the model's own" again; a new model always starts with its own."""
        if family not in ("sd15", "sdxl") or self.pipe is None or not hasattr(self.pipe, "vae"):
            return
        want = name or None
        st = getattr(self, "_vae_state", None)
        if not st or st[0] != id(self.pipe):  # (an id, not the pipeline itself, so an unloaded model can be freed)
            st = (id(self.pipe), None, self.pipe.vae)
        if st[1] == want:
            self._vae_state = st
            return
        orig = st[2]
        if want is None:
            self.pipe.vae = orig
        else:
            import vaes
            v = vaes.find(want)
            if v is None:
                raise UserError(f"The VAE “{want}” isn't in the VAE folder any more: pick another in Create.")
            if v["family"] != family:
                raise UserError(f"The VAE “{want}” is for {'SDXL' if v['family'] == 'sdxl' else 'SD 1.5'} models, and this "
                                f"is {'an SDXL' if family == 'sdxl' else 'an SD 1.5'} model: pick a matching VAE (or the model's own).")
            self.emit({"type": "status", "msg": f"Loading the VAE {want}…"})
            try:
                vae = vaes.build(v["file"], orig)
            except ValueError as e:
                raise UserError(str(e) + ".")
            self.pipe.vae = vae.to(device=orig.device, dtype=orig.dtype)
        # pipelines made from this one hold the old VAE: make them again when needed
        self.i2i_pipe = None
        self.inp_pipe = None
        self.cn_pipes = {}
        self._vae_state = (id(self.pipe), want, orig)

    def _i2i_pipe(self):
        if self.i2i_pipe is None:
            from diffusers import AutoPipelineForImage2Image
            self.i2i_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe, **_keep_dtype(self.pipe))  # shares weights
            self._quiet(self.i2i_pipe)
        return self.i2i_pipe

    def _inpaint_pipe(self):
        """Inpainting with the loaded model (works with normal checkpoints, not just inpainting ones)."""
        if self.inp_pipe is None:
            from diffusers import AutoPipelineForInpainting
            if "Inpaint" in type(self.pipe).__name__:
                return self.pipe  # an inpainting model is already an inpainting pipeline
            self.inp_pipe = AutoPipelineForInpainting.from_pipe(self.pipe, **_keep_dtype(self.pipe))
            self._quiet(self.inp_pipe)
        return self.inp_pipe

    # ------------------------------------------------------------ image prompt (IP-Adapter)
    IP_FILES = {
        # kind: (sd15 weight, sdxl weight). SDXL ones use the same ViT-H image encoder as SD 1.5.
        "style": ("ip-adapter_sd15.safetensors", "ip-adapter_sdxl_vit-h.safetensors"),
        "plus": ("ip-adapter-plus_sd15.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors"),
        "face": ("ip-adapter-plus-face_sd15.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
    }
    IP_REPO = "h94/IP-Adapter"

    def _set_ip_adapter(self, kind):
        """Load / switch / remove the image prompt adapter on the loaded SD 1.5 / SDXL model."""
        key = (self.loaded_key, kind) if kind else None
        if key == self.ip_key:
            return
        if self.ip_key is not None:
            try:
                self.pipe.unload_ip_adapter()
            except Exception as e:
                print("[engine] couldn't remove the image prompt adapter, reloading the model next time:", e, flush=True)
                self.loaded_key = None
            self.ip_key = None
        # pipelines made from this one copy its parts: rebuild them so they see the change
        self.i2i_pipe = None
        self.inp_pipe = None
        self.cn_pipes = {}
        self._tuning = None
        if not kind:
            self.release_memory()
            return
        sd15_w, sdxl_w = self.IP_FILES.get(kind, self.IP_FILES["style"])
        # Attention slicing from an earlier picture must go first: the adapter is built on top of the
        # current attention layers and can't be built on sliced ones ("SlicedAttnProcessor ... slice_size").
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        except Exception as e:
            print("[engine] couldn't reset attention before the image prompt adapter:", e, flush=True)
        enc_cached = _hf_file_cached(self.IP_REPO, "models/image_encoder/model.safetensors")
        self.emit({"type": "status", "msg": "Loading the image prompt adapter"
                                           + ("" if enc_cached else " (first time ~2.6 GB)") + "…"})
        with _quantized_unet_ok(self.pipe.unet):
            if self.family == "sdxl":
                self.pipe.load_ip_adapter(self.IP_REPO, subfolder="sdxl_models", weight_name=sdxl_w,
                                          image_encoder_folder="models/image_encoder")
            else:
                self.pipe.load_ip_adapter(self.IP_REPO, subfolder="models", weight_name=sd15_w)
        self.ip_key = key

    # ------------------------------------------------------------ Hi-res fix
    def _hires_fix(self, image, opts, text_kw, ip_image, img_seed, scheduler, steps, cfg, on_step):
        """Make it bigger, then redraw it at the bigger size (image to image) so the detail is real."""
        from PIL import Image
        scale = max(1.1, min(2.5, float(opts.get("scale", 1.5))))
        cap = _hires_cap(self.family)
        w, h = image.width * scale, image.height * scale
        if max(w, h) > cap:
            f = cap / max(w, h)
            w, h = w * f, h * f
        w, h = int(w) // 8 * 8, int(h) // 8 * 8
        if w <= image.width:
            return image, {"skipped": f"already at the largest size for this Mac ({cap}px)"}
        strength = max(0.1, min(0.9, float(opts.get("strength", 0.45))))
        hsteps = max(4, int(opts.get("steps") or max(8, steps // 2)))
        self.emit({"type": "status", "msg": f"Hi-res fix: redrawing at {w}×{h}…"})
        up = image.resize((w, h), Image.LANCZOS)
        pipe = self._i2i_pipe()
        if scheduler is not None:
            pipe.scheduler = scheduler
        self._tune_for_size(pipe, w, h)
        t = self.torch()
        call = dict(text_kw, image=up, strength=strength, num_inference_steps=int(math.ceil(hsteps / strength)),
                    guidance_scale=cfg, generator=t.Generator("cpu").manual_seed(img_seed), callback_on_step_end=on_step)
        if ip_image is not None:
            call["ip_adapter_image"] = ip_image
        out = pipe(**call).images[0]
        return out, {"scale": round(scale, 2), "strength": strength, "steps": hsteps, "size": [w, h]}

    # ------------------------------------------------------------ face detailer
    def _face_detail(self, image, opts, prompt, negative, text_args, ip_image, img_seed, scheduler, steps, cfg, on_step):
        import detailer
        if self.face_finder is None:
            self.emit({"type": "status", "msg": "Loading the face finder (first time ~100 MB)…"})
            self.face_finder = detailer.FaceFinder(paths.WEIGHTS_DIR / "facexlib")
        boxes = self.face_finder.find(image)[: max(1, int(opts.get("max_faces", 4)))]
        if not boxes:
            return image, {"faces": 0}
        work = 1024 if self.family == "sdxl" else 512
        strength = max(0.1, min(0.8, float(opts.get("strength", 0.4))))
        dsteps = max(6, int(opts.get("steps") or max(10, int(steps * 0.6))))
        face_prompt = (opts.get("prompt") or "").strip()
        kw = text_args(face_prompt or prompt, negative)
        pipe = self._inpaint_pipe()
        if scheduler is not None:
            pipe.scheduler = scheduler
        self._tune_for_size(pipe, work, work)
        t = self.torch()
        for k, box in enumerate(boxes):
            if self.cancel_event.is_set():
                raise Cancelled()
            self.emit({"type": "status", "msg": f"Face detailer: face {k + 1} of {len(boxes)}…"})
            crop, region, m = detailer.face_crop(image, box, padding=float(opts.get("padding", 1.8)), work=work)
            call = dict(kw, image=crop, mask_image=m, strength=strength, width=work, height=work,
                        num_inference_steps=int(math.ceil(dsteps / strength)), guidance_scale=cfg,
                        generator=t.Generator("cpu").manual_seed(img_seed + k + 1), callback_on_step_end=on_step)
            if ip_image is not None:
                call["ip_adapter_image"] = ip_image
            new = pipe(**call).images[0]
            image = detailer.paste_back(image, new, region, m)
        return image, {"faces": len(boxes), "strength": strength, "steps": dsteps,
                       **({"prompt": face_prompt} if face_prompt else {})}

    # ------------------------------------------------------------ comparison grid
    def compare(self, args, force=False, low_memory=False):
        """The same picture with different settings, side by side (plus each picture on its own)."""
        import grid
        xa, ya = args.get("x_axis"), args.get("y_axis") or None
        if xa not in grid.AXES or (ya and ya not in grid.AXES):
            raise UserError("Pick what to compare.")
        xs = grid.parse_values(xa, args.get("x_values"))
        ys = grid.parse_values(ya, args.get("y_values")) if ya else [None]
        if not xs or not ys:
            raise UserError("Enter the values to compare, separated by commas (e.g. 4, 6, 8).")
        if len(xs) * len(ys) > grid.MAX_CELLS:
            raise UserError(f"That's {len(xs) * len(ys)} pictures; the most is {grid.MAX_CELLS}. Use fewer values.")
        base = dict(args.get("base") or {})
        base["num_images"] = 1
        seed = base.get("seed")
        if seed in (None, "", -1) or (str(seed).lstrip("-").isdigit() and int(seed) < 0):
            base["seed"] = random.randint(0, 2**32 - 1)  # one seed for every cell, so only the setting changes
        lora_name = args.get("lora")
        images, records = {}, []
        total = len(xs) * len(ys)
        k = 0
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                if self.cancel_event.is_set():
                    raise Cancelled()
                k += 1
                cell = grid.apply(base, xa, x, lora_name)
                if ya:
                    cell = grid.apply(cell, ya, y, lora_name)
                cell["grid_cell"] = {"x": grid.label(xa, x), **({"y": grid.label(ya, y)} if ya else {})}
                self.emit({"type": "status", "msg": f"Comparison {k} of {total}: {grid.label(xa, x)}"
                                                   + (f", {grid.label(ya, y)}" if ya else "") + "…"})
                kept = []
                try:
                    recs = self.generate(cell, force=force, low_memory=low_memory, keep=kept)
                except UserError as e:
                    print(f"[engine] comparison cell failed ({cell['grid_cell']}): {e}", flush=True)
                    recs = []
                images[(xi, yi)] = kept[0] if kept else None
                records += recs
        title = (base.get("prompt") or "")[:120]
        out = grid.compose(images, xs, ys, xa, ya, title=title)
        settings = {"model": base.get("model"), "prompt": base.get("prompt", ""),
                    "negative_prompt": base.get("negative_prompt", ""), "seed": base["seed"],
                    "steps": base.get("steps"), "cfg": base.get("cfg"), "sampler": base.get("sampler") or "",
                    "loras": base.get("loras") or [], "width": out.width, "height": out.height,
                    "grid": {"x_axis": xa, "x_values": xs, "y_axis": ya, "y_values": ys if ya else None,
                             "images": [r["id"] for r in records]}}
        path = self._save(out, settings, suffix="grid")
        record = self.library.add_image(path, settings, kind="grid")
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
        record = self._content_check(record, out)
        self.emit({"type": "image", "record": record})
        return {"grid": record, "images": records}

    # ------------------------------------------------------------ pose map (OpenPose)
    def _get_pose_detector(self):
        if self.pose_detector is None:
            import pose
            self.emit({"type": "status", "msg": "Loading the pose finder"
                                               + ("" if self._is_cached(pose.POSE_REPO) else " (first time ~550 MB)") + "…"})
            self.pose_detector = pose.PoseDetector()
        return self.pose_detector

    # ------------------------------------------------------------ post-processing
    def get_face_restorer(self):
        if self.face_restorer is None:
            from postprocess.face_restore import FaceRestorer
            self.face_restorer = FaceRestorer(paths.WEIGHTS_DIR, self.device(), emit=self.emit)
        return self.face_restorer

    def _restore_faces(self, image, args):
        """Restore faces with the chosen model. Returns (image, info for the picture's settings)."""
        from postprocess.face_restore import model_key, model_label
        key = model_key(args.get("face_model"))
        fs = float(args.get("face_strength") if args.get("face_strength") is not None else 0.8)
        fid = float(args.get("face_fidelity") if args.get("face_fidelity") is not None else 0.5)
        self.emit({"type": "status", "msg": f"Restoring faces ({model_label(key)})…"})
        image, n_faces = self.get_face_restorer().restore(image, fs, model=key, fidelity=fid)
        info = {"model": key, "strength": fs, "faces": n_faces}
        if key == "codeformer":
            info["fidelity"] = fid
        return image, info

    def get_enhancer(self, force=False):
        if self.enhancer is None:
            from postprocess import finegrain
            ram = total_ram_gb()
            if ram and ram < finegrain.MIN_RAM_GB and not force:
                raise UserError(f"The enhancer needs about {finegrain.MIN_RAM_GB} GB of memory; this Mac has {ram} GB.")
            if not finegrain.is_cached():
                free = free_disk_gb(_hub_dir())
                if free < finegrain.DOWNLOAD_GB + 5:
                    raise UserError(f"The enhancer needs a ~{finegrain.DOWNLOAD_GB} GB download but only {free} GB is free.")
                self.emit({"type": "status", "msg": f"Downloading enhancer models (first time only, ~{finegrain.DOWNLOAD_GB} GB)…"})
            self.unload()  # free the generation model; only one big model at a time
            if 0 < ram < 16:
                self.drop_helpers()  # 8 GB Macs: nothing else may stay in memory next to the enhancer
            self.emit({"type": "status", "msg": "Loading Finegrain enhancer…"})
            self.enhancer = finegrain.FinegrainEnhancer(self.device())
        return self.enhancer

    def interrogate(self, args):
        """Image -> prompt. args: {path, mode: 'describe'|'quick'|'tags', detail, general_threshold}
        describe: Qwen3.5 4B (a modern vision model, Apple GPU) writes a detailed prompt; quick: Florence-2;
        tags: the WD tagger (Danbooru tags)."""
        from PIL import Image
        p = Path(str(args.get("path") or ""))
        if not p.is_file():
            raise UserError("Couldn't open that file.")
        try:
            import pictures
            p = Path(pictures.prepare(p))  # HEIC, TIFF, RAW… are converted by macOS first
        except Exception as e:
            raise UserError(str(e) or "Couldn't open that file.")
        try:
            image = Image.open(p)
            image.load()
        except Exception:
            raise UserError("That file isn't an image Protoart Studio can read.")
        mode = args.get("mode") or "describe"
        import vlm
        if mode == "describe" and vlm.supported():
            import net
            # the picture reader runs as its own process: free the image model first on small Macs
            if 0 < total_ram_gb() < 16:
                self.unload()
                self.prompt_enhancer = None
                self.enhancer = None
                self.interrogator = None
            try:
                text = vlm.describe(p, style="short" if args.get("detail") == "short" else "natural", emit=self.emit,
                                    cancel_event=self.cancel_event, allow_download=net.is_online())
            except vlm.Cancelled:
                raise Cancelled()
            except vlm.VlmError as e:
                raise UserError(str(e))
            if not text:
                raise UserError("The picture reader didn't write anything; try again or use Quick describe.")
            return {"prompt": text, "model": vlm.LABEL}
        if 0 < total_ram_gb() < 16 and mode != "tags":
            # Florence-2 + an image model + the app don't fit comfortably in 8 GB together.
            self.unload()
            self.prompt_enhancer = None
            self.enhancer = None
        if self.interrogator is None:
            from postprocess.interrogate import Interrogator
            self.interrogator = Interrogator(self.device(), emit=self.emit)
        if args.get("mode") == "tags":
            tags, rating = self.interrogator.tags(
                image, general_threshold=float(args.get("general_threshold", 0.35)),
                include_characters=bool(args.get("include_characters", True)))
            return {"prompt": ", ".join(tags), "tags": tags, "rating": rating}
        return {"prompt": self.interrogator.describe(image, detail=args.get("detail", "detailed"))}

    def find_text(self, args):
        """🔤 Text selection tool: the text in the picture (signs, logos, captions, watermarks) as a selection for the
        painting area -> {mask (PNG data URL, width × height), found, count}"""
        import base64
        import io
        from PIL import Image, ImageDraw, ImageFilter
        src, image = self._source_picture({"path": args.get("path"), "image_id": args.get("image_id")})
        image = image.convert("RGB")
        if 0 < total_ram_gb() < 16:
            self.unload()  # Florence-2 and an image model don't fit together on small Macs
        if self.interrogator is None:
            from postprocess.interrogate import Interrogator
            self.interrogator = Interrogator(self.device(), emit=self.emit)
        quads = self.interrogator.text_regions(image)
        W, H = int(args.get("width") or image.width), int(args.get("height") or image.height)
        sx, sy = W / image.width, H / image.height
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        for q in quads:
            pts = [(q[i] * sx, q[i + 1] * sy) for i in range(0, 8, 2)]
            d.polygon(pts, fill=255)
        if quads:  # a little past the letters (outlines, shadows, glow)
            grow = max(3, int(min(W, H) * 0.008)) // 2 * 2 + 1
            m = m.filter(ImageFilter.MaxFilter(min(grow, 15)))
        buf = io.BytesIO()
        m.save(buf, "PNG")
        return {"mask": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "found": bool(quads),
                "count": len(quads)}

    def postprocess(self, args, force=False):
        """args: {action: 'faces'|'enhance', image_id, ...options}. Saves a new linked library image."""
        from PIL import Image
        if args.get("image_id") is not None:
            src = self.library.get_image(int(args["image_id"]))
            if src is None:
                raise UserError("Image not found in the library.")
        else:
            src = self._external_source(args.get("path"))
        if not Path(src["path"]).exists():
            raise UserError(f"The image file is missing: {src['path']}")
        try:
            image = Image.open(src["path"]).convert("RGB")
        except Exception:
            raise UserError("That file isn't an image Protoart Studio can read.")
        settings = dict(src["settings"] or {})
        settings.update({k: src[k] for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        settings["loras"] = src["loras"]
        action = args.get("action")
        started = time.time()

        # "both" = enhance & upscale first, then restore faces on the sharper result (one saved image).
        steps = {"faces": ["faces"], "enhance": ["enhance"], "both": ["enhance", "faces"], "depth": ["depth"]}.get(action)
        if not steps:
            raise UserError(f"Unknown action: {action}")
        out = image
        for step in steps:
            if self.cancel_event.is_set():
                raise Cancelled()
            if step == "depth":
                import controlnet as cnet
                size = cnet.depth_size(args.get("depth_model"))
                style = "color" if args.get("style") == "color" else "gray"
                est = self._get_depth_estimator(size)
                self.emit({"type": "status", "msg": f"Making a depth map (Depth Anything V2 {size.capitalize()})…"})
                self.emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 1})
                out = cnet.depth_image(est, out, style)
                settings["depth"] = {"model": cnet.DEPTH_MODELS[size][0], "size": size, "style": style}
                continue
            if step == "enhance":
                enh = self.get_enhancer(force=force)
                opts = dict(args.get("options") or {})
                keep_style = bool(opts.pop("keep_style", False)) and not opts.get("instant")
                style_words = str(opts.pop("style_words", "") or "").strip()[:200]
                if not opts.get("prompt"):
                    # Reuse the image's own prompt so the enhancer adds the right kind of detail.
                    opts["prompt"] = "masterpiece, best quality, highres, " + (src["prompt"] or "")
                if keep_style:
                    # 🎨 Keep the picture's style: the enhancer's model is a photo model; without its "render" LoRA,
                    # with lighter changes, "photo" as a negative and the picture's own style words, it adds detail
                    # in the picture's own look (painting, drawing, anime) instead of turning it photographic
                    opts["prompt"] = ", ".join(x for x in ["best quality, highres", style_words, (src["prompt"] or "").strip()] if x)
                    opts["negative_prompt"] = ("photo, photograph, photorealistic, realistic, realistic skin, 3d render, cgi, "
                                               "worst quality, low quality, normal quality")
                    opts["lora_sdxl_render"] = 0.0
                    opts["lora_more_details"] = min(0.3, float(opts.get("lora_more_details", 0.5) or 0.5))
                    opts["denoise_strength"] = min(0.25, float(opts.get("denoise_strength", 0.35) or 0.35))
                    opts["controlnet_scale"] = max(0.8, float(opts.get("controlnet_scale", 0.6) or 0.6))
                before_enh = out
                seed = int(args.get("seed") if args.get("seed") not in (None, "", -1) else random.randint(0, 2**31 - 1))
                self.emit({"type": "status", "msg": f"Enhancing {out.width}×{out.height} → ×{opts.get('upscale_factor', 2)}…"})
                from postprocess.finegrain import EnhanceCancelled

                def prog(done, total):
                    self.wait_if_paused()  # Pause: hold between tiles
                    self.emit({"type": "progress", "image": 1, "images": 1, "step": done, "steps": total})
                last_pv = [0.0]

                def pv(lat):  # live preview while enhancing (every ~1.5 s), as in Create
                    if not self.previews or time.time() - last_pv[0] < 1.5:
                        return
                    last_pv[0] = time.time()
                    import preview
                    arr = lat[0].detach().float().cpu().numpy() if lat.dim() == 4 else lat.detach().float().cpu().numpy()
                    self.emit({"type": "preview", "image": 1, "src": preview.latent_to_data_url(arr, "sd15")})
                try:
                    out, used = enh.enhance(out, seed=seed, cancel_event=self.cancel_event, on_progress=prog,
                                            on_preview=pv, **opts)
                except EnhanceCancelled:
                    raise Cancelled()
                if keep_style:
                    try:  # the colours and light come back from the picture; the new detail stays
                        import color_match
                        out = color_match.keep_tones(out, before_enh.resize(out.size, Image.LANCZOS))
                    except Exception as e:
                        print("[engine] keep tones skipped:", e, flush=True)
                settings["enhance"] = dict(used, seed=seed, keep_style=keep_style, **({"style_words": style_words} if style_words else {}))
            else:
                if len(steps) > 1 and 0 < total_ram_gb() < 16:
                    self.enhancer = None  # 8 GB Macs: free the enhancer before the face model
                    self.release_memory()
                restored, info = self._restore_faces(out, {"face_strength": args.get("strength", 0.8),
                                                           "face_model": args.get("face_model"),
                                                           "face_fidelity": args.get("face_fidelity")})
                if info["faces"] == 0:
                    if len(steps) == 1:
                        raise UserError("No faces were found in this image.")
                    self.emit({"type": "status", "msg": "No faces found; saved the enhanced image only"})
                else:
                    out = restored
                settings["face_restore"] = info
        kind = {"faces": "faces", "enhance": "enhanced", "both": "enhanced", "depth": "depth"}[action]

        settings.update(width=out.width, height=out.height, seconds=round(time.time() - started, 1))
        path = self._save(out, settings, suffix=kind)
        record = self.library.add_image(path, settings, parent_id=src["id"], kind=kind)
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
        record = self._content_check(record, out)
        self.emit({"type": "image", "record": record})
        return record

    @staticmethod
    def _external_source(path):
        """Treat any image file (e.g. dropped on the Tools tab) like a library record."""
        p = Path(str(path or ""))
        if not p.is_file():
            raise UserError("Couldn't open that file.")
        try:
            import pictures
            p = Path(pictures.prepare(p))  # HEIC, TIFF, RAW… are converted by macOS first
        except Exception as e:
            raise UserError(str(e) or "Couldn't open that file.")
        settings = {}
        try:
            from PIL import Image
            with Image.open(p) as im:
                raw = (getattr(im, "text", None) or {}).get("beestudio")
                if not raw:  # JPEGs saved by this app keep the settings in the comment field
                    c = im.info.get("comment") or b""
                    c = c.decode("utf-8", "replace") if isinstance(c, bytes) else str(c)
                    if "\nbeestudio:" in c:
                        raw = c.split("\nbeestudio:", 1)[1]
            settings = json.loads(raw) if raw else {}
        except Exception:
            pass
        return {"id": None, "path": str(p), "settings": settings, "loras": settings.get("loras", []),
                "prompt": settings.get("prompt", ""), "negative_prompt": settings.get("negative_prompt", ""),
                "model": settings.get("model", "imported"), "sampler": settings.get("sampler", ""),
                "steps": settings.get("steps"), "cfg": settings.get("cfg"), "seed": settings.get("seed", 0)}

    # ------------------------------------------------------------ 🩹 Remove (spot healing) and ✒️ Vectorize
    def _source_picture(self, args, what="picture"):
        """The picture a tool works on: a Library picture (image_id) or any file (path)."""
        from PIL import Image, ImageOps
        if args.get("image_id") is not None:
            src = self.library.get_image(int(args["image_id"]))
            if src is None:
                raise UserError("Image not found in the library.")
        else:
            src = self._external_source(args.get("path") or args.get("init_image"))
        if not Path(src["path"]).exists():
            raise UserError(f"The image file is missing: {src['path']}")
        import pictures
        try:
            image = ImageOps.exif_transpose(Image.open(pictures.prepare(src["path"])))
        except Exception:
            raise UserError(f"That file isn't a {what} Protoart Studio can read.")
        return src, image

    _cut_cache = {}  # picture path + its time -> the matte (the last two pictures), so refining doesn't rerun the model

    def _cut_matte(self, src, image, emit):
        import net
        import postprocess.cutout as cut
        key = (src["path"], os.path.getmtime(src["path"]) if os.path.exists(src["path"]) else 0, image.size)
        if key in self._cut_cache:
            return self._cut_cache[key]
        if not cut.downloaded():
            if net.strict:
                raise UserError(f"Cut out needs a one-time download of its model (BiRefNet, ~{cut.MB} MB), and Strict "
                                "offline mode is on. Turn it off in Settings → Privacy.")
            emit({"type": "status", "msg": f"Downloading the Cut out model (BiRefNet, ~{cut.MB} MB, first time only)…"})
            net.set_online(True)
            try:
                sess = cut.session(download=True)
            finally:
                net.set_online(False)
        else:
            sess = cut.session(download=False)
        emit({"type": "status", "msg": "Finding the subject…"})
        m = cut.matte(image, sess)
        if len(self._cut_cache) >= 2:
            self._cut_cache.pop(next(iter(self._cut_cache)))
        self._cut_cache[key] = m
        return m

    def cutout(self, args, emit=None):
        """✂ Cut out (remove the background). args["mode"]:
        "matte": the model's matte as a working PNG (the window refines it live) -> {matte, width, height}
        "mask":  the subject as a selection for the painting area (👤 Subject tool) -> {mask, found}
        "save":  the refined cut-out saved to the Library (see-through PNG, or on a colour) -> the record"""
        import base64
        import numpy as np
        from PIL import Image
        import postprocess.cutout as cut
        emit = emit or self.emit
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        mode = args.get("mode") or "save"
        started = time.time()
        emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 1})
        m = self._cut_matte(src, image, emit)
        emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 1})
        if mode == "mask":
            png = cut.selection_png(m, args.get("width") or image.width, args.get("height") or image.height)
            return {"mask": "data:image/png;base64," + base64.b64encode(png).decode(), "found": bool((m > 0.5).any())}
        if mode == "matte":
            tmp = paths.HOME / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            for old in tmp.glob("cutout_matte_*.png"):
                try:
                    if time.time() - old.stat().st_mtime > 86400:
                        old.unlink()
                except OSError:
                    pass
            p = tmp / f"cutout_matte_{os.getpid()}_{int(time.time() * 1000)}.png"
            Image.fromarray(np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8), "L").save(p)
            return {"matte": str(p), "width": image.width, "height": image.height, "found": bool((m > 0.5).any())}
        if mode == "layers":  # 🗂 Split into layers: the subject (see-through) and the background with it filled in
            return self._cut_layers(src, image, m, args, emit, started)
        opts = {k: args.get(k) for k in ("threshold", "softness", "grow")}
        alpha = cut.refine(m, threshold=float(opts["threshold"] if opts["threshold"] is not None else 0.5),
                           softness=float(opts["softness"] if opts["softness"] is not None else 0.1),
                           grow=int(opts["grow"] or 0))
        if args.get("clean", True) is not False:
            alpha = cut.clean_bits(alpha)
        if not (alpha > 0.02).any():
            raise UserError("Nothing is left with these settings: lower the threshold, or grow the edge.")
        out = cut.compose(image, alpha, args.get("background"), fix_edges=args.get("fix_edges", True) is not False)
        if args.get("crop"):  # trim the see-through border around the subject
            ys, xs = np.nonzero(alpha > 0.02)
            pad = max(4, round(max(image.size) * 0.01))
            box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(image.width, xs.max() + 1 + pad), min(image.height, ys.max() + 1 + pad))
            out = out.crop(box)
        settings = dict(src.get("settings") or {})
        settings.update({k: src.get(k) for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        settings["loras"] = src.get("loras") or []
        settings.update(width=out.width, height=out.height, seconds=round(time.time() - started, 1),
                        cutout={**opts, "background": args.get("background") or "transparent", "crop": bool(args.get("crop"))})
        path = self._save(out, settings, suffix="cutout")
        record = self.library.add_image(path, settings, parent_id=src.get("id"), kind="cutout")
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
        emit({"type": "image", "record": record})
        return record

    def procreate_layers(self, args, emit=None):
        """A Procreate file's layers, each as its own see-through PNG in the Library (top layer first), plus the
        finished picture. The .procreate file is only read."""
        import procreate
        emit = emit or self.emit
        path = str(args.get("path") or "")
        if not path.lower().endswith(".procreate") or not os.path.isfile(path):
            raise UserError("Pick a .procreate file.")
        emit({"type": "status", "msg": "Reading the Procreate layers…"})
        emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 1})
        try:
            flat, doc = procreate.flattened(path)
            layers, _ = procreate.layers(path, include_hidden=bool(args.get("hidden")))
        except procreate.ProcreateError as e:
            raise UserError(str(e))
        records = []
        total = len(layers) + 1
        names = {}
        for i, (info, img) in enumerate([(None, flat)] + layers):
            if self.cancel_event.is_set():
                raise Cancelled()
            label = "flattened" if info is None else info["name"]
            names[label] = names.get(label, 0) + 1
            if info is not None and names[label] > 1:
                label = f"{label} ({names[label]})"
            settings = {"prompt": f"{doc['name']} · {label}", "width": img.width, "height": img.height, "seed": 0,
                        "procreate": {"file": os.path.basename(path), "layer": label,
                                      **({k: info[k] for k in ("opacity", "hidden", "clipped")} if info else {}),
                                      **({"blend": procreate.BLEND_NAMES.get(info["blend"], str(info["blend"]))} if info else {}),
                                      "position": i}}
            out = self._save(img, settings, suffix="layer" if info else "procreate")
            rec = self.library.add_image(out, settings, kind="layer" if info else "edited")
            rec["thumb"] = str(thumbs.make(rec["id"], out, img) or "") or None
            records.append(rec)
            emit({"type": "image", "record": rec})
            emit({"type": "progress", "image": 1, "images": 1, "step": i + 1, "steps": total})
        return {"records": records, "name": doc["name"], "layers": len(layers)}

    def _cut_layers(self, src, image, m, args, emit, started):
        import net
        import numpy as np
        from PIL import Image, ImageFilter
        import postprocess.cutout as cut
        import postprocess.heal as heal
        alpha = cut.refine(m, threshold=float(args.get("threshold") if args.get("threshold") is not None else 0.5),
                           softness=float(args.get("softness") if args.get("softness") is not None else 0.1),
                           grow=int(args.get("grow") or 0))
        alpha = cut.clean_bits(alpha)
        if not (alpha > 0.02).any():
            raise UserError("No subject was found with these settings: lower the threshold.")
        subject = cut.compose(image, alpha, None, fix_edges=True)
        # the background: what was behind the subject filled in (LaMa), a little past its edge
        mask = Image.fromarray(((alpha > 0.05) * 255).astype(np.uint8), "L").filter(
            ImageFilter.MaxFilter(max(3, (round(max(image.size) / 128) // 2) * 2 + 1)))
        emit({"type": "status", "msg": "Filling in the background behind the subject…"})
        if not heal.lama_downloaded() and not net.is_online():
            raise UserError("offline: the Smart fill model isn't on this Mac yet")
        sess = heal.lama_session(download=not heal.lama_downloaded())
        background = heal.heal(image, mask, method="lama", session=sess)
        base = dict(src.get("settings") or {})
        base.update({k: src.get(k) for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        records = []
        for label, img in (("subject", subject), ("background", background)):
            st = dict(base, width=img.width, height=img.height, seconds=round(time.time() - started, 1),
                      layer={"part": label, "of": os.path.basename(src["path"])})
            out = self._save(img, st, suffix=label)
            rec = self.library.add_image(out, st, parent_id=src.get("id"), kind="layer")
            rec["thumb"] = str(thumbs.make(rec["id"], out, img) or "") or None
            records.append(rec)
            emit({"type": "image", "record": rec})
        return {"records": records}

    def papercut(self, args, emit=None):
        """📄 Paper cut-out: the subject as a torn-paper sticker (border, shadow, print look, background)."""
        import postprocess.cutout as cut
        import postprocess.papercut as paper
        emit = emit or self.emit
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        started = time.time()
        emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 2})
        m = self._cut_matte(src, image, emit)
        emit({"type": "status", "msg": "Cutting it out of paper…"})
        emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 2})
        alpha = cut.refine(m, threshold=float(args.get("threshold") or 0.5), softness=0.08, grow=int(args.get("grow") or 0))
        alpha = cut.clean_bits(alpha)
        opts = {"edge": float(args.get("edge") if args.get("edge") is not None else 2.0),
                "edge_colour": args.get("edge_colour") or "white",
                "rough": float(args.get("rough") if args.get("rough") is not None else 0.5),
                "shadow": float(args.get("shadow") if args.get("shadow") is not None else 0.6),
                "print_as": args.get("print_as") or "photo",
                "background": args.get("background") or "transparent"}
        try:
            out = paper.make(image, alpha, **opts, seed=int(args.get("seed") or 7))
        except ValueError as e:
            raise UserError(str(e))
        emit({"type": "progress", "image": 1, "images": 1, "step": 2, "steps": 2})
        settings = dict(src.get("settings") or {})
        settings.update({k: src.get(k) for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        settings["loras"] = src.get("loras") or []
        settings.update(width=out.width, height=out.height, seconds=round(time.time() - started, 1), papercut=opts)
        path = self._save(out, settings, suffix="papercut")
        record = self.library.add_image(path, settings, parent_id=src.get("id"), kind="papercut")
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
        emit({"type": "image", "record": record})
        return record

    def heal(self, args, emit=None):
        """🩹 Remove: fill the painted area from its surroundings (LaMa, or OpenCV's quick fill)."""
        import detailer
        import net
        import postprocess.heal as heal
        emit = emit or self.emit  # Tools' side lane passes its own (so it doesn't take over the running job's progress)
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        try:
            mask = detailer.decode_mask(args.get("mask"), image.size)
        except Exception:
            raise UserError("Couldn't read the painted area.")
        if not mask.getbbox():
            raise UserError("Paint over what you want to remove first.")
        grow = int(args.get("grow") or 0)
        if grow > 0:  # 🪡 Puppet warp's background: cover the subject's edge fully, so no outline is left behind
            from PIL import ImageFilter
            mask = mask.convert("L").point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(min(61, grow * 2 + 1)))
        method = "quick" if args.get("method") == "quick" else "lama"
        started = time.time()
        emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 1})
        sess = None
        if method == "lama":
            if not heal.lama_downloaded():
                if net.strict:
                    raise UserError("Remove needs a one-time download of the LaMa model (~208 MB), and Strict offline mode "
                                    "is on. Turn it off in Settings → Privacy, or pick Quick fill.")
                emit({"type": "status", "msg": f"Downloading the Remove model (LaMa, ~{heal.LAMA_MB} MB, first time only)…"})
                net.set_online(True)
                try:
                    sess = heal.lama_session(download=True)
                finally:
                    net.set_online(False)
            else:
                sess = heal.lama_session(download=False)
        emit({"type": "status", "msg": "Removing what you painted…"})
        try:
            out = heal.heal(image, mask, method=method, session=sess)
        except ValueError as e:
            raise UserError(str(e))
        emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 1})
        if args.get("temp"):  # a working picture (Puppet warp's filled background): not saved to the Library
            tmp = paths.HOME / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            for old in tmp.glob("puppet_bg_*.png"):  # yesterday's working pictures
                try:
                    if time.time() - old.stat().st_mtime > 86400:
                        old.unlink()
                except OSError:
                    pass
            p = tmp / f"puppet_bg_{os.getpid()}_{int(time.time() * 1000)}.png"
            out.save(p)
            return {"path": str(p), "temp": True, "width": out.width, "height": out.height}
        settings = dict(src.get("settings") or {})
        settings.update({k: src.get(k) for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        settings["loras"] = src.get("loras") or []
        settings.update(width=out.width, height=out.height, seconds=round(time.time() - started, 1),
                        heal={"method": method})
        path = self._save(out, settings, suffix="healed")
        record = self.library.add_image(path, settings, parent_id=src.get("id"), kind="healed")
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
        emit({"type": "image", "record": record})
        return record

    EDIT_MODEL = "flux2_klein_mlx_q4"
    # ↔ Expand with FLUX.2 Klein: fal's outpaint LoRA (Apache-2.0, 76 MB). The picture goes in with a pure green
    # border where the new area should be, and this prompt asks the model to fill it.
    OUTPAINT_LORA = {"repo": "fal/flux-2-klein-4B-outpaint-lora", "revision": "b11770ac6a3cf9325dcf81742c12b1c4e257880f",
                     "file": "flux-outpaint-lora.safetensors", "mb": 76, "scale": 1.1,
                     "prompt": "Fill the green spaces according to the image"}

    def _outpaint_lora_file(self):
        """The outpaint LoRA on this Mac, downloaded once (from its publisher on Hugging Face, pinned version)."""
        import net
        from huggingface_hub import hf_hub_download
        L = self.OUTPAINT_LORA
        try:
            return hf_hub_download(L["repo"], L["file"], revision=L["revision"], local_files_only=True)
        except Exception:
            pass
        if net.strict:
            raise UserError(f"Expand with FLUX.2 needs a one-time download of the outpaint LoRA (~{L['mb']} MB), and "
                            "Strict offline mode is on. Turn it off in Settings → Privacy, or expand with the model at "
                            "the top instead.")
        self.emit({"type": "status", "msg": f"Downloading the outpaint LoRA (~{L['mb']} MB, first time only)…"})
        net.set_online(True)
        try:
            return hf_hub_download(L["repo"], L["file"], revision=L["revision"])
        finally:
            net.set_online(False)

    @staticmethod
    def green_canvas(image, left=0, right=0, top=0, bottom=0):
        """The picture on a bigger canvas whose new area is pure green (#00FF00), as the outpaint LoRA expects.
        -> (canvas, box where the original sits)"""
        from PIL import Image
        w, h = image.size
        canvas = Image.new("RGB", (w + left + right, h + top + bottom), (0, 255, 0))
        canvas.paste(image.convert("RGB"), (left, top))
        return canvas, (left, top, left + w, top + h)

    @staticmethod
    def align_to_original(result, original, box, max_shift=64):
        """FLUX.2 may redraw the scene a few pixels off. Find how far its version of your picture moved (phase
        correlation on a small grey copy), move the whole result back by that much, and say how different the
        picture area still is (0 = identical, 1 = unrelated; above ~0.35 it was redrawn). -> (aligned result, (dx, dy), difference)"""
        from PIL import Image
        import numpy as np
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        f = min(1.0, 256 / max(w, h))
        sw, sh = max(8, int(w * f)), max(8, int(h * f))
        a = np.asarray(original.convert("L").resize((sw, sh), Image.BILINEAR), dtype=np.float32)
        b = np.asarray(result.crop(box).convert("L").resize((sw, sh), Image.BILINEAR), dtype=np.float32)
        a -= a.mean()
        b -= b.mean()
        win = np.outer(np.hanning(sh), np.hanning(sw)).astype(np.float32)
        A, B = np.fft.fft2(a * win), np.fft.fft2(b * win)
        r = A * np.conj(B)
        r /= np.abs(r) + 1e-6
        corr = np.fft.ifft2(r).real
        py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
        if py > sh // 2:
            py -= sh
        if px > sw // 2:
            px -= sw
        dx, dy = int(round(px / f)), int(round(py / f))  # how far to move the result so it lines up
        if abs(dx) > max_shift or abs(dy) > max_shift:
            dx, dy = 0, 0
        if dx or dy:
            arr = np.asarray(result.convert("RGB"))
            pad = ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0)), (0, 0))
            arr = np.pad(arr, pad, mode="edge")
            H, W = result.height, result.width
            # take the window that puts the shifted content back in place
            oy, ox = (0 if dy >= 0 else -dy), (0 if dx >= 0 else -dx)
            arr = arr[oy:oy + H, ox:ox + W]
            result = Image.fromarray(np.ascontiguousarray(arr))
        # how alike the two versions of your picture are, shapes not just brightness: 1 − correlation
        o = np.asarray(original.convert("L").resize((sw, sh), Image.BILINEAR), dtype=np.float32)
        n = np.asarray(result.crop(box).convert("L").resize((sw, sh), Image.BILINEAR), dtype=np.float32)
        o, n = o - o.mean(), n - n.mean()
        ncc = float((o * n).sum() / (np.sqrt((o * o).sum() * (n * n).sum()) + 1e-6))
        return result, (dx, dy), max(0.0, 1.0 - ncc)

    def outpaint_klein(self, args):
        """↔ Expand with FLUX.2 Klein + the outpaint LoRA: works for any picture, whatever model made it."""
        import mlx_engine
        from PIL import Image, ImageDraw, ImageFilter
        spec = all_models().get(self.EDIT_MODEL)
        if spec is None:
            raise UserError("Expand with FLUX.2 needs the FLUX.2 Klein model, which isn't in this version.")
        o = args.get("outpaint") or {}
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        w, h = image.size
        pads = {k: max(0, int(o.get(k) or 0)) for k in ("left", "right", "top", "bottom")}
        if not any(pads.values()):
            raise UserError("Choose how far to expand on at least one side.")
        if max(pads["left"] + pads["right"], pads["top"] + pads["bottom"]) > 3 * max(w, h):
            raise UserError("That's a very big expansion: expand in smaller steps (up to about half the picture each time works best).")
        canvas, box = self.green_canvas(image, **pads)
        extra = (args.get("prompt") or "").strip()
        prompt = self.OUTPAINT_LORA["prompt"] + (f". {extra}" if extra else "")
        ram = total_ram_gb()
        side = 768 if 0 < ram < 12 else 1024
        scale = min(1.0, side / max(1.0, (canvas.width * canvas.height) ** 0.5))
        width = max(256, int(canvas.width * scale) // 16 * 16)
        height = max(256, int(canvas.height * scale) // 16 * 16)
        steps = max(1, min(12, int(args.get("steps") or spec["defaults"]["steps"])))
        seed = int(args.get("seed") if args.get("seed") not in (None, "", -1) else random.randint(0, 2**32 - 1)) % 2**32
        started = time.time()
        if self.cancel_event.is_set():
            raise Cancelled()
        lora = self._outpaint_lora_file()
        L = self.OUTPAINT_LORA
        try:
            self._mlx_load(spec, [], variant="edit", extra_loras=[(lora, L["scale"])])
            self.loaded_key = None
            self.family = spec["family"]
            tmp = paths.HOME / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            first = tmp / f"expand_src_{os.getpid()}.png"
            # the reference at exactly the output size (multiples of 16, under 1 MP): the model's copy of your
            # picture then lands on the same pixels (mflux would otherwise centre-crop it on its own)
            canvas.resize((width, height), Image.LANCZOS).save(first)
            out = tmp / f"expand_out_{os.getpid()}.png"
            self.emit({"type": "status", "msg": "Expanding with FLUX.2 Klein + outpaint LoRA (fast engine)…"})

            def on_event(ev):
                if ev.get("type") == "progress":
                    self.emit({"type": "progress", "image": 1, "images": 1, "step": ev["step"], "steps": ev["steps"]})
                elif ev.get("type") == "preview":
                    self.emit({"type": "preview", "image": 1, "src": ev["src"]})
                elif ev.get("type") == "status":
                    self.emit(ev)

            try:
                self.mlx.request({"cmd": "generate", "prompt": prompt, "negative_prompt": None, "width": width,
                                  "height": height, "steps": steps, "cfg": 1.0, "seed": seed, "out": str(out),
                                  "preview": self.previews, "image_paths": [str(first)]}, on_event=on_event)
                result = Image.open(out).convert("RGB")
            finally:
                for f in (out, first):
                    try:
                        Path(f).unlink()
                    except OSError:
                        pass
        except mlx_engine.CancelledError:
            raise Cancelled()
        except mlx_engine.EngineError as e:
            raise UserError(str(e))
        result = result.resize(canvas.size, Image.LANCZOS)
        result, shift, diff = self.align_to_original(result, image, box)
        print(f"[expand] lined up by {shift}, picture area differs {diff:.3f}", flush=True)
        if diff > 0.35:
            # FLUX.2 redrew your picture rather than keeping it: pasting the original back would leave seams and
            # doubled objects, so its whole picture is kept as it is
            secs = time.time() - started
            settings = {"model": self.EDIT_MODEL, "prompt": prompt, "negative_prompt": "", "width": result.width,
                        "height": result.height, "steps": steps, "cfg": 1.0, "sampler": "flow_euler", "seed": seed,
                        "engine": "mlx", "loras": [{"name": "flux-2-klein-4B-outpaint-lora", "weight": L["scale"]}],
                        "seconds": round(secs, 1), "inpaint": {"outpaint": pads, "engine": "flux2_outpaint_lora",
                                                               "kept_original": False}}
            path = self._save(result, settings, suffix="expanded")
            record = self.library.add_image(path, settings, parent_id=src.get("id"), kind="outpaint")
            record["thumb"] = str(thumbs.make(record["id"], path, result) or "") or None
            record = self._content_check(record, result)
            self.emit({"type": "status", "msg": "FLUX.2 redrew the picture a little, so its whole version was kept "
                                                "(pasting your original back would leave seams). Try another seed, or "
                                                "expand less at a time, to keep the original exactly."})
            self.emit({"type": "image", "record": record})
            return record
        # your picture stays exactly as it was: only the new area comes from the model, blended over a band wide
        # enough to hide the join
        feather = max(6, min(40, min(w, h) // 24))
        keep = Image.new("L", canvas.size, 0)
        x0, y0, x1, y1 = box
        ImageDraw.Draw(keep).rectangle([x0 + (feather if pads["left"] else 0), y0 + (feather if pads["top"] else 0),
                                        x1 - 1 - (feather if pads["right"] else 0), y1 - 1 - (feather if pads["bottom"] else 0)],
                                       fill=255)
        keep = keep.filter(ImageFilter.GaussianBlur(feather / 2))
        orig = canvas.copy()
        result = Image.composite(orig, result, keep)
        secs = time.time() - started
        print(f"[timing] expand (FLUX.2 outpaint LoRA): {secs:.1f}s for {steps} steps at {width}x{height}", flush=True)
        settings = {"model": self.EDIT_MODEL, "prompt": prompt, "negative_prompt": "", "width": result.width,
                    "height": result.height, "steps": steps, "cfg": 1.0, "sampler": "flow_euler", "seed": seed,
                    "engine": "mlx", "loras": [{"name": "flux-2-klein-4B-outpaint-lora", "weight": L["scale"]}],
                    "seconds": round(secs, 1), "inpaint": {"outpaint": pads, "engine": "flux2_outpaint_lora",
                                                           "kept_original": True, "aligned": list(shift)}}
        path = self._save(result, settings, suffix="expanded")
        record = self.library.add_image(path, settings, parent_id=src.get("id"), kind="outpaint")
        record["thumb"] = str(thumbs.make(record["id"], path, result) or "") or None
        record = self._content_check(record, result)
        self.emit({"type": "image", "record": record})
        return record

    def _klein_edit(self, image, prompt, seed, steps=None):
        """FLUX.2 Klein edits one picture as described (fast engine) -> PIL image the same size. Used by regions."""
        import mlx_engine
        from PIL import Image
        spec = all_models().get(self.EDIT_MODEL)
        if spec is None:
            raise UserError("This needs the FLUX.2 Klein model, which isn't in this version.")
        steps = max(1, min(12, int(steps or spec["defaults"]["steps"])))
        width, height = max(256, image.width // 16 * 16), max(256, image.height // 16 * 16)
        tmp = paths.HOME / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        src = tmp / f"kedit_src_{os.getpid()}.png"
        out = tmp / f"kedit_out_{os.getpid()}.png"
        image.convert("RGB").save(src)
        try:
            self._mlx_load(spec, [], variant="edit")
            self.loaded_key = None
            self.family = spec["family"]

            def on_event(ev):
                if ev.get("type") == "progress":
                    self.emit({"type": "progress", "image": 1, "images": 1, "step": ev["step"], "steps": ev["steps"]})

            self.mlx.request({"cmd": "generate", "prompt": prompt, "negative_prompt": None, "width": width, "height": height,
                              "steps": steps, "cfg": 1.0, "seed": int(seed) % 2**32, "out": str(out), "preview": False,
                              "image_paths": [str(src)]}, on_event=on_event)
            return Image.open(out).convert("RGB").resize(image.size, Image.LANCZOS)
        except mlx_engine.CancelledError:
            raise Cancelled()
        except mlx_engine.EngineError as e:
            raise UserError(str(e))
        finally:
            for f in (src, out):
                try:
                    f.unlink()
                except OSError:
                    pass

    def edit_words(self, args):
        """✨ Edit with words (Image to image): FLUX.2 [klein] changes the picture as described. Optional: a painted
        area (only that part changes; the rest stays exactly as it was) and more pictures to take a style, an object
        or an outfit from."""
        if isinstance(args.get("outpaint"), dict):  # ↔ Expand with FLUX.2 (same queue entry as Edit with words)
            return self.outpaint_klein(args)
        import detailer
        import mlx_engine
        from PIL import Image, ImageFilter
        spec = all_models().get(self.EDIT_MODEL)
        if spec is None:
            raise UserError("Edit with words needs the FLUX.2 Klein model, which isn't in this version.")
        prompt = (args.get("prompt") or "").strip()
        if not prompt and not isinstance(args.get("combine"), dict):
            raise UserError("Describe the change in the prompt, e.g. “make the jacket red” or “turn it into a watercolour”.")
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        mask = None
        if args.get("mask"):
            try:
                mask = detailer.decode_mask(args.get("mask"), image.size)
            except Exception:
                raise UserError("Couldn't read the painted area.")
            if not mask.getbbox():
                mask = None
        refs = [r for r in (args.get("refs") or []) if isinstance(r, str) and r][:3]
        combine = args.get("combine") if isinstance(args.get("combine"), dict) else None
        ref_imgs = []
        import pictures
        for r in refs:
            try:
                from PIL import ImageOps
                ref_imgs.append(ImageOps.exif_transpose(Image.open(pictures.prepare(r))))
            except Exception:
                raise UserError(f"Couldn't read the extra picture: {os.path.basename(r)}")
        if combine is not None:  # 🧩 Combine: the scene is picture 1, the characters / things to add are the others
            image, ref_imgs, mask, prompt = self._combine_prepare(image, refs, ref_imgs, combine, prompt)
        ram = total_ram_gb()
        side = 768 if 0 < ram < 12 else 1024
        # small pictures are worked on bigger (up to 2×): faces and small details come out much closer to the original
        scale = min(2.0, side / max(1.0, (image.width * image.height) ** 0.5))
        width = max(256, int(image.width * scale) // 16 * 16)
        height = max(256, int(image.height * scale) // 16 * 16)
        steps = max(1, min(12, int(args.get("steps") or spec["defaults"]["steps"])))
        keep_faces = bool(args.get("keep_faces"))
        faces_redone = 0
        panels_cropped = False
        seed = int(args.get("seed") if args.get("seed") not in (None, "", -1) else random.randint(0, 2**32 - 1)) % 2**32
        started = time.time()
        if self.cancel_event.is_set():
            raise Cancelled()
        try:
            # LoRAs for FLUX.2 (e.g. a model chain step with its own LoRA); Edit with words itself sends none
            self._mlx_load(spec, [l for l in (args.get("loras") or []) if isinstance(l, dict) and l.get("name")], variant="edit")
            self.loaded_key = None
            self.family = spec["family"]
            tmp = paths.HOME / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            first = tmp / f"edit_src_{os.getpid()}.png"
            image.save(first)
            ref_files = [str(first)]
            for i, im in enumerate(ref_imgs):
                im = im.convert("RGB")
                p = tmp / f"edit_ref_{os.getpid()}_{i}.png"
                im.save(p)
                ref_files.append(str(p))
            out = tmp / f"edit_out_{os.getpid()}.png"
            self.emit({"type": "status", "msg": "Editing with FLUX.2 Klein (fast engine)…"})
            face_files = []

            def on_event(ev):
                if ev.get("type") == "progress":
                    self.emit({"type": "progress", "image": 1, "images": 1, "step": ev["step"], "steps": ev["steps"]})
                elif ev.get("type") == "preview":
                    self.emit({"type": "preview", "image": 1, "src": ev["src"]})
                elif ev.get("type") == "status":
                    self.emit(ev)

            try:
                self.mlx.request({"cmd": "generate", "prompt": prompt, "negative_prompt": None, "width": width,
                                  "height": height, "steps": steps, "cfg": 1.0, "seed": seed, "out": str(out),
                                  "preview": self.previews, "image_paths": ref_files}, on_event=on_event)
                result = Image.open(out).convert("RGB")
                cam = args.get("camera") if isinstance(args.get("camera"), dict) else None
                if cam and (cam.get("layout") or "single") == "single" and not self.cancel_event.is_set():
                    # 📷 Klein sometimes answers a camera move with a split / comic-panel picture: try again once with
                    # another seed and firmer words, and if it's still split, keep the biggest panel
                    import postprocess.compose as comp
                    if comp.panel_seams(result):
                        self.emit({"type": "status", "msg": "It came out split into panels: trying once more as one picture…"})
                        seed = (seed + 7919) % 2**32
                        self.mlx.request({"cmd": "generate", "prompt": prompt + " The whole frame is one single photograph "
                                          "of one moment: one person shown once, no grid, no split, no second view.",
                                          "negative_prompt": None, "width": width, "height": height, "steps": steps,
                                          "cfg": 1.0, "seed": seed, "out": str(out), "preview": self.previews,
                                          "image_paths": ref_files}, on_event=on_event)
                        result = Image.open(out).convert("RGB")
                        if comp.panel_seams(result):
                            result = result.crop(comp.largest_panel(result))
                            panels_cropped = True
                if keep_faces and mask is None and combine is None and not self.cancel_event.is_set():
                    # 🙂 Keep faces: each face is edited again on its own, close up (it fills the picture, so Klein
                    # keeps its features), with the original face as a second picture, and blended into the result
                    result, faces_redone = self._klein_faces(image, result, prompt, seed, steps, tmp, on_event, face_files)
            finally:
                for f in [out, *map(Path, ref_files), *face_files]:
                    try:
                        Path(f).unlink()
                    except OSError:
                        pass
        except mlx_engine.CancelledError:
            raise Cancelled()
        except mlx_engine.EngineError as e:
            raise UserError(str(e))
        # a small picture worked on bigger keeps the bigger result (unless only a painted part changed)
        if panels_cropped:
            pass  # the kept panel has its own shape: not stretched back to the picture's
        elif mask is not None or width * height < image.width * image.height * 1.3:
            result = result.resize(image.size, Image.LANCZOS)
        if mask is not None:
            # only the painted area changes, with a soft edge; everything else is the untouched original
            m = mask.convert("L").point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(5))
            m = m.filter(ImageFilter.GaussianBlur(max(2, min(image.size) // 200)))
            try:  # the redraw often comes out a little lighter overall: match its light to the original around it
                import color_match
                result = color_match.match_to_original(result, image, m)
            except Exception as e:
                print("[engine] colour match skipped:", e, flush=True)
            result = Image.composite(result, image, m)
        secs = time.time() - started
        print(f"[timing] edit with words: {secs:.1f}s for {steps} steps at {width}x{height}", flush=True)
        settings = {"model": self.EDIT_MODEL, "prompt": prompt, "negative_prompt": "", "width": result.width,
                    "height": result.height, "steps": steps, "cfg": 1.0, "sampler": "flow_euler", "seed": seed,
                    "engine": "mlx", "loras": [], "seconds": round(secs, 1),
                    "edit": {"painted": mask is not None, "extra_pictures": len(refs), "faces_redone": faces_redone}}
        if isinstance(args.get("camera"), dict):  # 📷 Camera angle
            settings["camera"] = {k: args["camera"].get(k) for k in ("az", "el", "zoom", "view", "layout")}
            settings["camera"]["panel_cropped"] = panels_cropped
        if combine is not None:
            settings["combine"] = {"placed": bool(combine.get("place")), "cut_out": combine.get("cutout") is not False,
                                   "match_light": combine.get("match_light") is not False}
        if args.get("loras"):
            settings["loras"] = [{"name": l["name"], "weight": float(l.get("weight", 1.0))} for l in args["loras"] if isinstance(l, dict) and l.get("name")]
        kind = "combined" if combine is not None else "camera_angle" if isinstance(args.get("camera"), dict) else "word_edit"
        path = self._save(result, settings, suffix={"combined": "combined", "camera_angle": "angle"}.get(kind, "wordedit"))
        record = self.library.add_image(path, settings, parent_id=src.get("id"), kind=kind)
        record["thumb"] = str(thumbs.make(record["id"], path, result) or "") or None
        record = self._content_check(record, result)
        self.emit({"type": "image", "record": record})
        return record

    def _klein_faces(self, original, result, prompt, seed, steps, tmp, on_event, files, max_faces=3):
        """🙂 Keep faces (Edit with words): find the faces in the original; for each, the same edit is made again on a
        close-up crop of the original face (with that face as a reference picture too), then blended into the result
        with its light matched. -> (result, how many faces were redone)"""
        import detailer
        from PIL import Image
        base = original.resize(result.size, Image.LANCZOS)
        ff = self._face_finder_ready()
        if ff is None:
            self.emit({"type": "status", "msg": "🙂 Keep faces needs the face finder that Restore faces downloads: use Restore faces once (Tools), then it works offline."})
            return result, 0
        try:
            boxes = ff.find(base, min_score=0.6, min_size=12)[:max_faces]
        except Exception as e:
            print("[engine] face finder unavailable:", e, flush=True)
            return result, 0
        work = 768 if 0 < total_ram_gb() < 12 else 1024
        done = 0
        for i, box in enumerate(boxes):
            if self.cancel_event.is_set():
                break
            self.emit({"type": "status", "msg": f"Keeping the face close to the original ({i + 1} of {len(boxes)})…"})
            crop, region, fmask = detailer.face_crop(base, box, padding=2.2, work=work)
            src_p, out_p = tmp / f"edit_face_{os.getpid()}_{i}.png", tmp / f"edit_face_out_{os.getpid()}_{i}.png"
            crop.save(src_p)
            files += [src_p, out_p]
            text = (prompt + " This is a close-up of the face: keep exactly the same person, with the same face shape, "
                    "eyes, nose, mouth, age, expression and head angle as in the pictures; change only what the "
                    "instruction asks for.")
            self.mlx.request({"cmd": "generate", "prompt": text, "negative_prompt": None, "width": work, "height": work,
                              "steps": steps, "cfg": 1.0, "seed": seed, "out": str(out_p), "preview": False,
                              "image_paths": [str(src_p), str(src_p)]}, on_event=on_event)
            new = Image.open(out_p).convert("RGB")
            left, top, right, bottom = region
            around = result.crop(region).resize(new.size, Image.LANCZOS)
            try:  # the close-up's light and colour matched to the result around the face, so it blends in
                import color_match
                new = color_match.match_to_original(new, around, fmask)
            except Exception as e:
                print("[engine] face colour match skipped:", e, flush=True)
            result = detailer.paste_back(result, new, region, fmask)
            done += 1
        return result, done

    def _combine_prepare(self, image, refs, ref_imgs, combine, prompt):
        """🧩 Combine: the characters (or things) cut out of their pictures, their light matched to the scene, and
        (when placed by hand) put where they go, so Klein only blends them in.
        -> (picture 1 for Klein, the reference pictures, the area that may change or None, the prompt)"""
        import numpy as np
        from PIL import Image
        import postprocess.compose as comp
        if not ref_imgs:
            raise UserError("Add the character (or thing) to put into the scene: drop its picture under “Add to the scene”.")
        cutout = combine.get("cutout") is not False
        light = combine.get("match_light") is not False
        strength = float(combine.get("light_strength") if combine.get("light_strength") is not None else 0.7)
        boxes = combine.get("place") if isinstance(combine.get("place"), list) else None
        pieces, refs_out = [], []
        for i, (path, im) in enumerate(zip(refs, ref_imgs)):
            rgba = im.convert("RGBA")
            if cutout:
                self.emit({"type": "status", "msg": f"Cutting out picture {i + 2} (finding the character)…"})
                import postprocess.cutout as cut
                m = self._cut_matte({"path": path}, im.convert("RGB"), self.emit)
                alpha = cut.clean_bits(cut.refine(m, threshold=0.5, softness=0.1, grow=0))
                if (alpha > 0.05).any():
                    rgba = cut.compose(im.convert("RGB"), alpha, None, fix_edges=True)
            box = boxes[i] if boxes and i < len(boxes) and isinstance(boxes[i], dict) else None
            if light:
                region = None
                if box:
                    l, t, w, h = comp.fit_box(rgba.size, box, image.size)
                    region = (l - w, t - h // 2, l + 2 * w, t + h + h // 2)  # the scene around where it goes
                rgba = comp.match_light(rgba, image, strength, region)
            if box:
                pieces.append((rgba, box))
            # the reference Klein sees: the cut-out on white, cropped to it (no old background to copy)
            a = np.asarray(rgba.getchannel("A"), np.float32) / 255.0
            bb = comp.subject_bbox(a) if cutout else None
            ref = comp.on_white(rgba.crop(bb) if bb else rgba) if cutout else rgba.convert("RGB")
            refs_out.append(ref)
        mask = None
        if pieces:
            image, cover = comp.place(image, pieces)
            if combine.get("keep_scene") is not False:
                mask = comp.blend_area(cover)
        prompt = comp.combine_prompt(prompt, len(refs_out), placed=bool(pieces))
        return image, refs_out, mask, prompt

    def peel(self, args, emit=None):
        """🧅 Peel layers: the painted / selected part lifted off as its own see-through layer, and what was behind
        it filled in (LaMa), so the next part can be peeled from what's left. Working files, not in the Library
        until saved. -> {"layer", "background"} (paths)"""
        import detailer
        import net
        import postprocess.compose as comp
        import postprocess.heal as heal
        emit = emit or self.emit
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        try:
            mask = detailer.decode_mask(args.get("mask"), image.size)
        except Exception:
            raise UserError("Couldn't read the painted area.")
        if not mask.getbbox():
            raise UserError("Paint over or select the part to peel off first.")
        emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 2})
        matte = None
        if args.get("tidy"):
            import postprocess.cutout as cut
            m = self._cut_matte(src, image, emit)
            matte = cut.refine(m, threshold=0.5, softness=0.1, grow=0)
        try:
            layer, hole = comp.peel(image, mask, matte=matte)
        except ValueError as e:
            raise UserError(str(e))
        emit({"type": "status", "msg": "Filling in what was behind it…"})
        if args.get("fill") == "none":
            background = image
        else:
            if not heal.lama_downloaded():
                if net.strict:
                    raise UserError("Peel layers fills in behind each layer with the LaMa model (~208 MB, a one-time "
                                    "download), and Strict offline mode is on. Turn it off in Settings → Privacy.")
                emit({"type": "status", "msg": f"Downloading the fill model (LaMa, ~{heal.LAMA_MB} MB, first time only)…"})
                net.set_online(True)
                try:
                    sess = heal.lama_session(download=True)
                finally:
                    net.set_online(False)
            else:
                sess = heal.lama_session(download=False)
            background = heal.heal(image, hole, method="lama", session=sess)
        emit({"type": "progress", "image": 1, "images": 1, "step": 2, "steps": 2})
        tmp = paths.HOME / "tmp" / "peel"
        tmp.mkdir(parents=True, exist_ok=True)
        for old in tmp.glob("*.png"):  # working pictures from days ago
            try:
                if time.time() - old.stat().st_mtime > 3 * 86400:
                    old.unlink()
            except OSError:
                pass
        stamp = f"{os.getpid()}_{int(time.time() * 1000)}"
        lp, bp = tmp / f"layer_{stamp}.png", tmp / f"rest_{stamp}.png"
        layer.save(lp)
        background.save(bp)
        return {"layer": str(lp), "background": str(bp), "width": image.width, "height": image.height}

    def peel_save(self, args, emit=None):
        """🧅 Save the peeled layers: a layered Photoshop file (.psd) next to the pictures, and each layer in the
        Library as a see-through PNG. args: {"original", "background", "layers": [{"path", "name", "visible"}]
        (top first)}"""
        from PIL import Image
        import postprocess.compose as comp
        emit = emit or self.emit
        src, original = self._source_picture({"init_image": args.get("original")})
        items = [l for l in (args.get("layers") or []) if isinstance(l, dict) and l.get("path")]
        if not items:
            raise UserError("Peel off at least one layer first.")
        tmp_root = (paths.HOME / "tmp" / "peel").resolve()

        def working(p):
            p = Path(p).resolve()
            if tmp_root not in p.parents:
                raise UserError("That layer isn't one of this app's working files.")
            if not p.exists():
                raise UserError("A peeled layer's working file is gone: peel it again.")
            return Image.open(p)

        size = original.size
        bg = working(args["background"]).convert("RGBA") if args.get("background") else original.convert("RGBA")
        stack = [("Background", bg.resize(size) if bg.size != size else bg, True)]
        for l in reversed(items):  # bottom first in the file
            im = working(l["path"]).convert("RGBA")
            stack.append((str(l.get("name") or "Layer")[:120], im.resize(size) if im.size != size else im,
                          l.get("visible") is not False))
        base = dict(src.get("settings") or {})
        base.update({k: src.get(k) for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        records = []
        for name, im, _vis in stack:
            st = dict(base, width=im.width, height=im.height, layer={"part": name, "of": os.path.basename(src["path"])})
            out = self._save(im if name != "Background" else im.convert("RGB"), st, suffix="layer")
            rec = self.library.add_image(out, st, parent_id=src.get("id"), kind="layer")
            rec["thumb"] = str(thumbs.make(rec["id"], out, im) or "") or None
            records.append(rec)
            emit({"type": "image", "record": rec})
        psd = Path(records[0]["path"]).with_name(Path(records[0]["path"]).stem.replace("_layer", "") + "_layers.psd")
        n = 2
        while psd.exists():
            psd = psd.with_name(f"{psd.stem}~{n}.psd")
            n += 1
        comp.write_psd(psd, size, stack)
        return {"records": records, "psd": str(psd), "layers": len(stack)}

    QUANT_BITS = {8: "int8", 6: "int6", 4: "uint4"}

    def quantize_model(self, args):
        """🗜 Make a smaller copy of an SD 1.5 / SDXL model: the UNet (most of its size) stored at 8, 6 or 4 bits
        with SDNQ, saved as a new model folder. Uses less memory and disk; pictures look nearly the same
        (8-bit: no visible change, 6-bit: very close, 4-bit: a little softer). It isn't faster per step."""
        import sdnq
        model_id = args.get("model")
        bits = int(args.get("bits") or 8)
        if bits not in self.QUANT_BITS:
            raise UserError("Pick 8, 6 or 4 bits.")
        spec = all_models().get(model_id) or {}
        if spec.get("family") not in ("sd15", "sdxl") or spec.get("engine") == "mlx" or spec.get("sdnq") or spec.get("quantize_unet"):
            raise UserError("Smaller copies can be made of SD 1.5 and SDXL models that aren't already reduced.")
        label = spec["label"].replace(" (your file)", "")
        dest = paths.QUANT_DIR / f"{_slug(label)}_{bits}bit"
        need = 2.6 if spec["family"] == "sdxl" else 1.4
        if free_disk_gb(paths.QUANT_DIR.parent) < need * 1.5 + 2:
            raise UserError(f"Not enough free disk space for the copy (~{need:.1f} GB).")
        phases = 4
        def phase(i, msg):
            if self.cancel_event.is_set():
                raise Cancelled()
            self.emit({"type": "status", "msg": msg})
            self.emit({"type": "progress", "image": 1, "images": 1, "step": i, "steps": phases})
        started = time.time()
        phase(0, f"Loading {label}…")
        self.load(model_id)
        pipe = self.pipe
        phase(1, f"Converting the UNet to {bits}-bit…")
        pipe.unet = sdnq.sdnq_post_load_quant(pipe.unet, weights_dtype=self.QUANT_BITS[bits], torch_dtype=pipe.unet.dtype)
        gc.collect()
        phase(2, f"Saving the {bits}-bit copy…")
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            pipe.save_pretrained(str(dest))
            shutil.rmtree(dest / "unet", ignore_errors=True)  # the reduced UNet is written in SDNQ's own format
            sdnq.save_sdnq_model(pipe.unet, str(dest / "unet"))
            (dest / "protoart_quant.json").write_text(json.dumps({"bits": bits, "weights_dtype": self.QUANT_BITS[bits],
                                                                  "family": spec["family"], "from": label}))
        except Exception as e:
            shutil.rmtree(dest, ignore_errors=True)
            raise UserError(f"Couldn't save the copy: {e}")
        finally:
            self.unload()  # the loaded model was changed in place: load it fresh next time
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
        phase(4, f"Saved {label} · {bits}-bit ({size:.1f} GB) in {time.time() - started:.0f} s")
        return {"path": str(dest), "label": f"{label} · {bits}-bit", "bits": bits, "gb": round(size, 2),
                "family": spec["family"]}

    def tiled_upscale(self, args):
        """🧩 Tiled upscale (MultiDiffusion): enlarge a picture and let the Create model (SD 1.5 / SDXL) redraw it
        lightly, tile by tile, adding real detail. Saved as a new picture."""
        import tiled
        src, image = self._source_picture(args)
        model_id = args.get("model")
        spec = all_models().get(model_id) or {}
        family = spec.get("family")
        if family not in ("sd15", "sdxl") or spec.get("engine") == "mlx" or spec.get("inpaint"):
            raise UserError("Tiled upscale uses an SD 1.5 or SDXL model: pick one in Create first.")
        scale = max(1.25, min(4.0, float(args.get("scale") or 2)))
        limit = 4096 if total_ram_gb() >= 16 else 3072
        if max(image.size) * scale > limit:
            scale = limit / max(image.size)
            if scale < 1.05:
                raise UserError(f"That picture is already about as big as this Mac can upscale ({limit} px).")
        self.load(model_id)
        pipe = self.pipe
        _set_seamless(pipe, False)
        tile_px = 1024 if family == "sdxl" else 512
        self._tune_for_size(pipe, tile_px, tile_px)  # attention slicing on small Macs (without it: 10× slower)
        sampler = args.get("sampler")
        if sampler:
            try:
                import samplers
                pipe.scheduler = samplers.build_scheduler(family, sampler, pipe.scheduler.config)
            except Exception:
                pass
        steps = int(args.get("steps") or 20)
        strength = max(0.1, min(0.6, float(args.get("strength") or 0.3)))
        cfg = float(args.get("cfg") or (6.0 if family == "sdxl" else 7.0))
        seed = int(args.get("seed") if args.get("seed") not in (None, "", -1) else random.randint(0, 2**31 - 1))
        # A detail prompt, not the picture's scene: each tile only sees part of the picture, and a scene prompt makes
        # every tile draw the whole scene again (extra faces and people). Your words are added for texture / style.
        extra = (args.get("prompt") or "").strip()
        prompt = ", ".join(x for x in (extra, "highly detailed, sharp focus, fine texture, high quality") if x)
        negative = args.get("negative_prompt") or "blurry, lowres, artifacts, extra faces, extra people, duplicate, watermark, text"
        started = time.time()

        def on_step(i, total):
            self.wait_if_paused()
            if self.cancel_event.is_set():
                raise Cancelled()
            self.emit({"type": "progress", "image": 1, "images": 1, "step": i + 1, "steps": total})

        out = tiled.upscale(pipe, family, image, prompt, negative, scale=scale, strength=strength, steps=steps, cfg=cfg,
                            seed=seed, on_step=on_step, emit=self.emit)
        settings = {"model": model_id, "prompt": prompt, "negative_prompt": negative, "width": out.width,
                    "height": out.height, "steps": steps, "cfg": cfg, "seed": seed, "strength": strength,
                    "tiled_upscale": {"scale": round(scale, 2)}, "seconds": round(time.time() - started, 1)}
        path = self._save(out, settings, suffix="tiled")
        record = self.library.add_image(path, settings, parent_id=src.get("id") if isinstance(src, dict) else None, kind="tiled")
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
        record = self._content_check(record, out)
        self.emit({"type": "image", "record": record})
        self.release_memory()
        return record

    def faceswap(self, args):
        """🔁 Face swap: the face from args["face"] onto the person (or people) in the picture (init_image / image_id).
        Only for faces you have permission to use; results that the picture checker finds explicit aren't saved."""
        import detailer
        import faceswap as fsw
        if not args.get("consent"):
            raise UserError("Tick “I have permission to use this face” first.")
        _, target = self._source_picture({"image_id": args.get("image_id"), "path": args.get("init_image") or args.get("path")})
        _, face_pic = self._source_picture({"path": args.get("face")}, "face picture")
        target, face_pic = target.convert("RGB"), face_pic.convert("RGB")
        if max(target.size) > 3072:  # very big photos: work on a copy that fits memory
            s = 3072 / max(target.size)
            target = target.resize((int(target.width * s), int(target.height * s)))
        if self.face_swapper is None:
            if self.face_finder is None:
                self.emit({"type": "status", "msg": "Preparing the face finder…"})
                self.face_finder = detailer.FaceFinder(paths.WEIGHTS_DIR / "facexlib")
            self.face_swapper = fsw.FaceSwapper(self.face_finder.net, emit=self.emit)
        started = time.time()
        try:
            image, n = self.face_swapper.swap(face_pic, target, which=args.get("which") or "all",
                                              source_index=args.get("face_index") or 0, boost=args.get("boost") or 1)
        except ValueError as e:
            msg = str(e)
            raise UserError(msg[:1].upper() + msg[1:] + ".")
        info = {"faces": n, "boost": int(args.get("boost") or 1), "licence": fsw.LICENCE}
        if args.get("restore", True):  # tidy the new face (CodeFormer, blended so skin texture survives)
            image, finfo = self._restore_faces(image, {"face_model": args.get("face_model") or "codeformer",
                                                       "face_strength": args.get("face_strength", 0.6),
                                                       "face_fidelity": args.get("face_fidelity", 0.7)})
            info["restore"] = finfo
        # safeguard: a face swap is never saved when the picture checker finds the result explicit
        try:
            score = self._get_nsfw_checker().score(image)
        except Exception as e:
            print("[engine] face swap check failed:", e, flush=True)
            raise UserError("Face swap needs the picture checker (it checks every result). Connect to the internet "
                            "once so it can download (~90 MB), then try again.")
        import nsfw
        if score >= nsfw.THRESHOLD:
            raise UserError("This face swap wasn't saved: the result looks explicit. Face swap only makes "
                            "non-explicit pictures.")
        settings = {"prompt": "face swap", "model": "inswapper_128", "width": image.width, "height": image.height,
                    "seed": 0, "faceswap": info, "seconds": round(time.time() - started, 1),
                    "licence": fsw.LICENCE}
        src_id = args.get("image_id")
        path = self._save(image, settings, suffix="faceswap")
        record = self.library.add_image(path, settings, parent_id=int(src_id) if src_id is not None else None, kind="faceswap")
        self.library.set_nsfw_score(record["id"], score)
        record["thumb"] = str(thumbs.make(record["id"], path, image) or "") or None
        self.emit({"type": "image", "record": record})
        return record

    def protect_merge(self, args):
        """Patchbay 🛡 Protect: put back the original picture wherever you painted, over the chain's result, with a
        soft edge. Saved as a new picture in the Library (the chain's result stays too)."""
        import detailer
        from PIL import Image, ImageFilter
        res = self.library.get_image(int(args["image_id"]))
        if res is None or not Path(res["path"]).exists():
            raise UserError("The result isn't in the Library any more.")
        result = Image.open(res["path"]).convert("RGB")
        _, original = self._source_picture({"path": args.get("original"), "image_id": args.get("original_id")})
        original = original.convert("RGB").resize(result.size, Image.LANCZOS)
        try:
            mask = detailer.decode_mask(args.get("mask"), result.size)
        except Exception:
            raise UserError("Couldn't read the protected area.")
        if not mask.getbbox():
            raise UserError("Paint over the areas to protect first.")
        soft = mask.convert("L").filter(ImageFilter.GaussianBlur(max(2, min(result.size) // 160)))
        merged = Image.composite(original, result, soft)
        settings = dict(res.get("settings") or {})
        settings["protected_areas"] = True
        path = self._save(merged, settings, suffix="protected")
        record = self.library.add_image(path, settings, parent_id=res.get("parent_id") or res.get("id"), kind=res.get("kind") or "edited")
        record["thumb"] = str(thumbs.make(record["id"], path, merged) or "") or None
        self.emit({"type": "image", "record": record})
        return record

    def vectorize(self, args):
        """✒️ Vectorize: an SVG of flat colour shapes, saved next to the pictures (not in the Library)."""
        import postprocess.vectorize as vec
        src, image = self._source_picture(args)
        colors = int(args.get("colors") or 8)
        detail = args.get("detail") if args.get("detail") in vec.DETAIL else "medium"
        self.emit({"type": "status", "msg": f"Tracing {colors} colours…"})
        self.emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 1})
        svg, stats = vec.vectorize(image, colors=colors, detail=detail, smooth=args.get("smooth", True) is not False)
        day = paths.IMAGES_DIR / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = Path(src["path"]).stem
        out = day / f"{stem}_vector.svg"
        n = 2
        while out.exists():
            out = day / f"{stem}_vector~{n}.svg"
            n += 1
        out.write_text(svg, encoding="utf-8")
        self.emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 1})
        self.emit({"type": "status", "msg": f"Vector saved: {out.name} ({stats['shapes']} shapes)"})
        return {"path": str(out), "source": src["path"], "source_id": src.get("id"), **stats,
                "colors_asked": colors, "detail": detail}

    def _save(self, image, settings, suffix=None):
        from PIL.PngImagePlugin import PngInfo
        day = paths.IMAGES_DIR / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        tail = f"_{suffix}" if suffix else ""
        fmt, quality = self.output_format
        if image.mode in ("RGBA", "LA"):  # see-through pictures (✂ Cut out) are always PNG: JPEG has no transparency
            fmt = "png"
        ext = "jpg" if fmt == "jpeg" else "png"
        path = day / f"{datetime.now().strftime('%H%M%S')}_{_slug(settings.get('prompt', ''))}_{settings.get('seed', 0)}{tail}.{ext}"
        n = 2
        while path.exists():  # never overwrite an earlier picture saved in the same second
            path = path.with_name(f"{path.stem.rsplit('~', 1)[0]}~{n}{path.suffix}")
            n += 1
        mode = getattr(self, "embed_meta", "details")
        if mode == "none":  # Settings: don't embed generation details in pictures
            if fmt == "jpeg":
                image.convert("RGB").save(path, "JPEG", quality=int(quality), optimize=True)
            else:
                image.save(path)
            return path
        # What goes into the file is what people see when a picture is shared: file paths on this Mac are
        # cut down to file names (the library keeps the full details).
        settings = _shareable(settings)
        meta = PngInfo()
        lora_txt = "".join(f" <lora:{l['name']}:{l['weight']}>" for l in settings.get("loras") or [])
        meta_text = (f"{settings.get('prompt', '')}{lora_txt}\nNegative prompt: {settings.get('negative_prompt', '')}\n"
                     f"Steps: {settings.get('steps')}, Sampler: {settings.get('sampler')}, CFG scale: {settings.get('cfg')}, "
                     f"Seed: {settings.get('seed')}, Size: {settings.get('width')}x{settings.get('height')}, "
                     f"Model: {settings.get('model')}"
                     + (f", Clip skip: {settings['clip_skip']}" if settings.get("clip_skip") else ""))
        meta.add_text("parameters", meta_text)
        meta.add_text("beestudio", json.dumps(settings))
        if fmt == "jpeg":
            # JPEG has no PNG text chunks: keep the same info in the JPEG comment field.
            comment = meta_text + "\nbeestudio:" + json.dumps(settings)
            image.convert("RGB").save(path, "JPEG", quality=int(quality), optimize=True,
                                      comment=comment.encode("utf-8"))
        else:
            image.save(path, pnginfo=meta)
        return path
