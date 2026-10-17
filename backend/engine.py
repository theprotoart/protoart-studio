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
        out["file:" + p.name] = {
            "label": f"{p.stem} ({'SDXL' if family == 'sdxl' else 'SD 1.5'} checkpoint)",
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
            out["tdict:" + Path(i["file"]).name] = {
                "label": f"{i['name']} (DiffusionBee)", "family": "sd15", "tdict": i["file"],
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
        try:
            m.setdefault(lm["id"], model_hub.local_to_spec(lm))
        except Exception as e:
            print("[engine] bad local model entry:", lm.get("path"), e, flush=True)
    for um in model_hub.USER_MODELS:  # base models added from Hugging Face (Settings -> Models)
        try:
            m[um["id"]] = model_hub.to_spec(um)
        except Exception as e:
            print("[engine] bad added model entry:", um, e, flush=True)
    for mid, fam in model_hub.MODEL_FAMILY.items():  # Settings → Fix a type: you picked SD 1.5 / SDXL for a file
        spec = m.get(mid)
        if spec and fam in ("sd15", "sdxl") and spec.get("family") != fam and (spec.get("file") or spec.get("tdict")):
            m[mid] = dict(spec, family=fam, detected_family=spec.get("family"), family_set=True,
                          defaults=dict(model_hub.DEFAULTS[fam]),
                          min_ram_gb=16 if fam == "sdxl" else 8, low_memory_min_ram_gb=8 if fam == "sdxl" else None)
        elif spec and spec.get("family") == fam:
            m[mid] = dict(spec, family_set=True)
    return m


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
        self.controlnets = {}       # ControlNet models by type (share the loaded SD 1.5 model)
        self.cn_pipes = {}          # (type, img2img) -> pipeline sharing the loaded model's weights
        self.depth_estimator = None
        self.inp_pipe = None        # inpainting pipeline sharing the loaded model's weights
        self.ip_key = None          # (model, kind) of the loaded image prompt adapter, or None
        self.face_finder = None     # RetinaFace for the face detailer
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
             "sizes": ERNIE_SIZES if v["family"] == "ernie" else None, "source": v.get("source"),
             "engine": v.get("engine", "torch"), "fast_mode": v["family"] in FAST_LORAS and not v.get("engine"),
             "clip_skip": recommended_clip_skip(v, k), "flavor": model_flavor(v, k),
             "path": (str(v.get("file") or v.get("tdict") or v.get("local_dir") or "") or None) if ":" in k else None,
             "family_set": bool(v.get("family_set")), "detected_family": v.get("detected_family"),
             "single_file": bool(v.get("file") or v.get("tdict")) and ":" in k,
             "nsfw_auto": nsfw_words.name_flag(v["label"], k, os.path.basename(
                 str(v.get("file") or v.get("tdict") or v.get("local_dir") or "")))}
            for k, v in all_models().items()
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
        if spec.get("engine") == "mlx":
            jobs.append(lambda: snapshot_download(spec["mlx_repo"]))
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
    def _is_cached(repo, with_pe=False, ignore_extra=()):
        """True if the model files are already downloaded (no network access)."""
        try:
            from huggingface_hub import snapshot_download
            ignore = list(ignore_extra) + ([] if with_pe else ["pe/*", "pe_tokenizer/*"])
            snapshot_download(repo, local_files_only=True, ignore_patterns=ignore or None)
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
        if self.local_copy(spec):
            return True
        if spec.get("file") or spec.get("tdict") or spec.get("local_dir"):
            return True
        if spec.get("engine") == "mlx":
            return self._is_cached(spec["mlx_repo"])
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
        if spec.get("local_dir"):  # a diffusers folder on this Mac: load it like a repo, never online
            spec = dict(spec, repo=spec["local_dir"])
        family = spec["family"]
        use_pe = bool(use_pe) and family == "ernie"
        # Low-memory (8-bit) only for SDXL: SD 1.5 already fits in 8 GB, and 8-bit loading of single-file
        # SD 1.5 models was a source of loading errors. ERNIE builds are already quantized.
        low_memory = bool(low_memory) and family == "sdxl"
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
                rcp = local_configs.ensure(family)  # offline model recipe (configs + tokenizer)
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
                import local_configs
                rcp = local_configs.ensure(family)  # your own checkpoint: load fully offline
                fkw = {"config": rcp, "local_files_only": True}
                if family == "sd15":
                    fkw.update(safety_checker=None, requires_safety_checker=False)
                if low_memory:
                    unet = self._file_unet_8bit(spec, dtype, rcp)
                    pipe = cls.from_single_file(spec["file"], torch_dtype=dtype, unet=unet, **fkw)
                else:
                    pipe = cls.from_single_file(spec["file"], torch_dtype=dtype, **fkw)
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
        self.fuse_ok = family in ("sd15", "sdxl") and not low_memory and not spec.get("quantize_unet") \
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
        try:
            return trainer.train(self, args, self.emit, self.cancel_event, paths.LORAS_DIR)
        except trainer._Cancelled:
            raise Cancelled()
        except trainer.TrainError as e:
            raise UserError(str(e))
        finally:
            self.unload()  # training modifies the loaded model; always start fresh afterwards

    def train_embedding(self, args):
        """Train an embedding (textual inversion) on SD 1.5: Train tab → Embedding."""
        import embeddings
        import ti_trainer
        import trainer
        if self.mlx is not None:
            self.mlx.stop()
        self.unload()
        self.drop_helpers()
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
        return {"captioned": done, "items": trainer.scan_dataset(args.get("folder"))}

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
        depth = self._get_depth_estimator(depth_model) if (kind == "depth" and not already_map) else None
        posed = self._get_pose_detector() if (kind == "openpose" and not already_map) else None
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
        if kind not in self.controlnets:
            if 0 < total_ram_gb() < 16:
                self.controlnets = {}  # 8 GB Macs: keep only one ControlNet loaded
                self.cn_pipes = {}
                self.release_memory()
            dtype = self.pipe.unet.dtype
            self.controlnets[kind] = cnet.load(kind, self.pipe.unet, dtype, emit=self.emit).to(self.device())
        key = (kind, img2img)
        if key not in self.cn_pipes:
            cls = diffusers.StableDiffusionControlNetImg2ImgPipeline if img2img else diffusers.StableDiffusionControlNetPipeline
            self.cn_pipes[key] = cls.from_pipe(self.pipe, controlnet=self.controlnets[kind])
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

    def _generate_mlx(self, args, model_id, spec, prompt, negative, width, height, steps, cfg, n, seed,
                      original_prompt, force=False):
        """ERNIE and FLUX on the fast engine (MLX worker process)."""
        import mlx_engine
        from PIL import Image
        fam = spec["family"]
        fam_name = {"ernie": "ERNIE", "flux": "FLUX"}.get(fam, fam)
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
                    self.mlx.request({"cmd": "generate", "prompt": p_i, "negative_prompt": n_i,
                                      "width": width, "height": height, "steps": steps, "cfg": cfg, "seed": img_seed,
                                      "out": str(out), "preview": self.previews}, on_event=on_event,
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
                path = self._save(image, settings)
                record = self.library.add_image(path, settings, kind="generated")
                record["thumb"] = str(thumbs.make(record["id"], path, image) or "") or None
                record = self._content_check(record, image)
                results.append(record)
                self.emit({"type": "image", "record": record})
            return results
        except mlx_engine.CancelledError:
            raise Cancelled()
        except mlx_engine.EngineError as e:
            raise UserError(str(e))

    def _mlx_load(self, spec, lora_reqs, allow_download=True):
        """Get a fast-engine (MLX) model ready in its worker process. Returns this Mac's RAM in GB.
        The worker keeps the model loaded; asking again for the same model and LoRAs costs nothing."""
        import mlx_engine
        import net
        fam = spec["family"]
        fam_name = {"ernie": "ERNIE", "flux": "FLUX"}.get(fam, fam)
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
            loras.append([info["file"], float(r.get("weight", 1.0))])
        if not mlx_engine.installed():
            if not net.is_online():
                raise UserError("offline: the fast engine isn't installed yet")  # retried online by the backend
        self.mlx.install()
        if self.cancel_event.is_set():
            raise Cancelled()
        load_started = time.time()
        self.mlx.request({"cmd": "load", "repo": spec["mlx_repo"], "base": spec.get("mlx_base"),
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
        slice_attn = self.family in ("sd15", "sdxl") and (ram < 64) and not self.ip_key
        # (attention slicing would replace the image prompt adapter's layers, so it stays off while one is loaded)
        tile_vae = (ram < 12 and mp > 0.7) or (ram < 24 and mp > 1.1) or mp > 2.5
        want = (slice_attn, tile_vae)
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
            # (never touched while an image prompt adapter is loaded: switching slicing on OR off swaps out
            # every attention layer, including the adapter's, and the next picture fails)
            call(["enable_attention_slicing"] if slice_attn else ["disable_attention_slicing"])
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
                                                  cancel_event=self.cancel_event, family=family)
        if self.cancel_event.is_set():
            raise Cancelled()
        return {"prompt": text, "original": prompt, "style": "tags" if family == "sd15" else "natural"}

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
        model_id = args["model"]
        models = all_models()
        if model_id not in models:
            raise UserError(f"Unknown model: {model_id}")
        spec = models[model_id]
        d = spec["defaults"]
        family = spec["family"]
        is_sd = family in ("sd15", "sdxl") and spec.get("engine") != "mlx"

        width = int(args.get("width") or d["width"])
        height = int(args.get("height") or d["height"])
        steps = int(args.get("steps") or d["steps"])
        cfg = float(args.get("cfg") if args.get("cfg") not in (None, "") else d["cfg"])
        n = max(1, min(int(args.get("num_images") or 1), 16))
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
            if not is_sd:
                raise UserError("Image-to-image works with SD 1.5 and SDXL for now (not ERNIE or FLUX).")
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
        if control:
            if family != "sd15":
                raise UserError("ControlNet works with SD 1.5 models (pick one in the model list).")
            if mask is not None:
                raise UserError("Inpainting/expanding and ControlNet can't be combined yet: switch one off.")
            import controlnet as cnet
            if control["type"] not in cnet.TYPES:
                raise UserError(f"Unknown ControlNet type: {control['type']}")
            src_path = args.get("control_image") or args.get("init_image")
            if not src_path:
                raise UserError("Add an image for the ControlNet to follow.")
            control_src = open_image(src_path, "ControlNet image")
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
        mult = 16 if family in ("ernie", "flux") else 8
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

        if spec.get("engine") == "mlx":
            return self._generate_mlx(args, model_id, spec, prompts, negatives, width, height, steps, cfg, n, seed,
                                      originals, force=force)

        lean = self.is_lean(spec)
        lean_enc = {}
        if lean:
            texts = sorted(set(prompts) | (set(negatives) if cfg > 1 else set()))
            for text, emb in zip(texts, self._ernie_embeddings(spec, texts)):
                lean_enc[text] = emb

        self.load(model_id, force=force, low_memory=low_memory)
        if self.cancel_event.is_set():  # stopped while the model was loading
            raise Cancelled()
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
                return kw

            call = dict(num_inference_steps=steps, guidance_scale=cfg, generator=gen, callback_on_step_end=on_step)
            if is_sd:
                call.update(text_args(prompt, negative))
            if ip_image is not None:
                call["ip_adapter_image"] = ip_image
            if control_map is not None:
                call.update(controlnet_conditioning_scale=float(control.get("scale", 1.0)),
                            control_guidance_start=float(control.get("start", 0.0)),
                            control_guidance_end=float(control.get("end", 1.0)))
                if init_image is not None:
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
                    # keep everything outside the mask exactly as it was (the model slightly changes it)
                    image = Image.composite(image, init_image, mask)
                if hires:
                    phase["name"] = "Hi-res fix"
                    image, extras_info["hires"] = self._hires_fix(image, hires, text_args(prompt, negative), ip_image,
                                                                  img_seed, scheduler, steps, cfg, on_step)
                    phase["name"] = ""
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
                **({"clip_skip": clip_skip} if is_sd else {}),
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
            record = self._content_check(record, image)
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

    def _i2i_pipe(self):
        if self.i2i_pipe is None:
            from diffusers import AutoPipelineForImage2Image
            self.i2i_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)  # shares weights: no extra memory
            self._quiet(self.i2i_pipe)
        return self.i2i_pipe

    def _inpaint_pipe(self):
        """Inpainting with the loaded model (works with normal checkpoints, not just inpainting ones)."""
        if self.inp_pipe is None:
            from diffusers import AutoPipelineForInpainting
            self.inp_pipe = AutoPipelineForInpainting.from_pipe(self.pipe)
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
        """Image -> prompt. args: {path, mode: 'describe'|'tags', detail, general_threshold}"""
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
        if 0 < total_ram_gb() < 16 and args.get("mode") != "tags":
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
                if not opts.get("prompt"):
                    # Reuse the image's own prompt so the enhancer adds the right kind of detail.
                    opts["prompt"] = "masterpiece, best quality, highres, " + (src["prompt"] or "")
                seed = int(args.get("seed") if args.get("seed") not in (None, "", -1) else random.randint(0, 2**31 - 1))
                self.emit({"type": "status", "msg": f"Enhancing {out.width}×{out.height} → ×{opts.get('upscale_factor', 2)}…"})
                from postprocess.finegrain import EnhanceCancelled

                def prog(done, total):
                    self.wait_if_paused()  # Pause: hold between tiles
                    self.emit({"type": "progress", "image": 1, "images": 1, "step": done, "steps": total})
                try:
                    out, used = enh.enhance(out, seed=seed, cancel_event=self.cancel_event, on_progress=prog, **opts)
                except EnhanceCancelled:
                    raise Cancelled()
                settings["enhance"] = dict(used, seed=seed)
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

    def heal(self, args):
        """🩹 Remove: fill the painted area from its surroundings (LaMa, or OpenCV's quick fill)."""
        import detailer
        import net
        import postprocess.heal as heal
        src, image = self._source_picture(args)
        image = image.convert("RGB")
        try:
            mask = detailer.decode_mask(args.get("mask"), image.size)
        except Exception:
            raise UserError("Couldn't read the painted area.")
        if not mask.getbbox():
            raise UserError("Paint over what you want to remove first.")
        method = "quick" if args.get("method") == "quick" else "lama"
        started = time.time()
        self.emit({"type": "progress", "image": 1, "images": 1, "step": 0, "steps": 1})
        sess = None
        if method == "lama":
            if not heal.lama_downloaded():
                if net.strict:
                    raise UserError("Remove needs a one-time download of the LaMa model (~208 MB), and Strict offline mode "
                                    "is on. Turn it off in Settings → Privacy, or pick Quick fill.")
                self.emit({"type": "status", "msg": f"Downloading the Remove model (LaMa, ~{heal.LAMA_MB} MB, first time only)…"})
                net.set_online(True)
                try:
                    sess = heal.lama_session(download=True)
                finally:
                    net.set_online(False)
            else:
                sess = heal.lama_session(download=False)
        self.emit({"type": "status", "msg": "Removing what you painted…"})
        try:
            out = heal.heal(image, mask, method=method, session=sess)
        except ValueError as e:
            raise UserError(str(e))
        self.emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 1})
        settings = dict(src.get("settings") or {})
        settings.update({k: src.get(k) for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")})
        settings["loras"] = src.get("loras") or []
        settings.update(width=out.width, height=out.height, seconds=round(time.time() - started, 1),
                        heal={"method": method})
        path = self._save(out, settings, suffix="healed")
        record = self.library.add_image(path, settings, parent_id=src.get("id"), kind="healed")
        record["thumb"] = str(thumbs.make(record["id"], path, out) or "") or None
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
        lora_txt = "".join(f" <lora:{l['name']}:{l['weight']}>" for l in settings["loras"])
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
