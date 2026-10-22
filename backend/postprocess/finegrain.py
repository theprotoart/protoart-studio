"""Finegrain Image Enhancer (Clarity-style creative upscaler).

Same models and defaults as https://github.com/peanutcocktail/Finegrain-Image-Enhancer:
4x-UltraSharp ESRGAN pre-upscale, then tiled re-diffusion with Juggernaut Reborn (SD 1.5)
guided by a tile ControlNet, plus the "more details" and "SDXL render" LoRAs.
Runs through finegrain's `refiners` library. Total first-time download ≈ 3.5 GB.
"""
from pathlib import Path

DOWNLOAD_GB = 3.5
MIN_RAM_GB = 8

# (repo, filename, pinned revision) exactly as used by the original app
_FILES = {
    "unet": ("refiners/juggernaut.reborn.sd1_5.unet", "model.safetensors", "347d14c3c782c4959cc4d1bb1e336d19f7dda4d2"),
    "clip_text_encoder": ("refiners/juggernaut.reborn.sd1_5.text_encoder", "model.safetensors", "744ad6a5c0437ec02ad826df9f6ede102bb27481"),
    "lda": ("refiners/juggernaut.reborn.sd1_5.autoencoder", "model.safetensors", "3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19"),
    "controlnet_tile": ("refiners/controlnet.sd1_5.tile", "model.safetensors", "48ced6ff8bfa873a8976fa467c3629a240643387"),
    "esrgan": ("philz1337x/upscaler", "4x-UltraSharp.pth", "011deacac8270114eb7d2eeff4fe6fa9a837be70"),
    "negative_embedding": ("philz1337x/embeddings", "JuggernautNegative-neg.pt", "203caa7e9cc2bc225031a4021f6ab1ded283454a"),
}
_LORAS = {
    "more_details": ("philz1337x/loras", "more_details.safetensors", "a3802c0280c0d00c2ab18d37454a8744c44e474e"),
    "sdxl_render": ("philz1337x/loras", "SDXLrender_v2.0.safetensors", "a3802c0280c0d00c2ab18d37454a8744c44e474e"),
}

SOLVERS = ["DDIM", "DPMSolver", "Euler"]
# Turbo: the distilled LCM-LoRA for SD 1.5 (latent-consistency/lcm-lora-sdv1-5, OpenRAIL++, ~135 MB, downloaded
# the first time Turbo is used). Few steps and no negative-prompt pass: a fraction of the work per tile.
TURBO_LORA = ("latent-consistency/lcm-lora-sdv1-5", "pytorch_lora_weights.safetensors")
TURBO_SCALE = 8 / 64      # its alpha / rank
TURBO_MAX_STEPS = 8
# refiners first resizes every picture to this short side (its `downscale_size`). 8 GB Macs work at 512 px
# for the quick speeds (ProtoArt Beautify's 8 GB profile): less than half the work, and no swapping.
WORK_SHORT_SIDE = 768
SMALL_MAC_WORK_SIDE = 512


def memory_profile(ram_gb=None):
    """Settings that keep each Mac out of swap (measured on an 8 GB M1 in ProtoArt Beautify)."""
    ram = _ram_gb() if ram_gb is None else ram_gb
    if 0 < ram < 16:
        return {"name": "8 GB", "tile_tokens": 9216, "vae_tile": 512, "esrgan_tile": 192, "esrgan_overlap": 16,
                "slice": 1024, "low_memory": True}
    if 0 < ram < 24:
        return {"name": "16 GB", "tile_tokens": 12288, "vae_tile": 1024, "esrgan_tile": 256, "esrgan_overlap": 16,
                "slice": 2048, "low_memory": True}
    return {"name": "24 GB+", "tile_tokens": 16384, "vae_tile": 1024, "esrgan_tile": 512, "esrgan_overlap": 32,
            "slice": None, "low_memory": False}


def work_side(o, ram_gb=None):
    """The short side the picture is worked at: "auto" = 512 px for Instant / Turbo / Fast on Macs with less
    than 16 GB, else 768 px; or 512 / 768 when chosen."""
    ws = o.get("work_size")
    if ws in (512, 768, "512", "768"):
        return int(ws)
    ram = _ram_gb() if ram_gb is None else ram_gb
    quick = o.get("turbo") or o.get("instant") or int(o.get("num_inference_steps") or 18) <= 10
    return SMALL_MAC_WORK_SIDE if (0 < ram < 16 and quick) else WORK_SHORT_SIDE

DEFAULTS = {
    "upscale_factor": 2, "prompt": "masterpiece, best quality, highres",
    "negative_prompt": "worst quality, low quality, normal quality",
    "denoise_strength": 0.35,   # "creativity"
    "controlnet_scale": 0.6,    # "resemblance"
    "controlnet_decay": 1.0, "condition_scale": 6, "tile_width": 112, "tile_height": 144,
    "num_inference_steps": 18, "solver": "DDIM",
    "lora_more_details": 0.5, "lora_sdxl_render": 1.0,
    "turbo": False,
    "instant": False,           # 4x-UltraSharp only: sharpens in seconds, invents no new detail
    "work_size": "auto",        # short side the picture is worked at (see work_side)
}


def _work_dims(w, h, side):
    """refiners' resize_modulo_8: short side -> `side`, both sides multiples of 8 (rounded down)."""
    scale = side / (max(1, min(w, h)) * 8)
    return int(w * scale) * 8, int(h * scale) * 8


def output_size(w, h, factor, side=WORK_SHORT_SIDE):
    """The size the enhancer really produces: the picture is first resized so its short side is `side`
    (768 px, or 512 on small Macs for the quick speeds), then upscaled by `factor`."""
    ww, wh = _work_dims(w, h, side)
    return int(ww * factor), int(wh * factor)


def is_cached():
    from huggingface_hub import try_to_load_from_cache
    for repo, name, rev in list(_FILES.values()) + list(_LORAS.values()):
        hit = try_to_load_from_cache(repo, name, revision=rev)
        if not isinstance(hit, str):
            return False
    return True


class FinegrainEnhancer:
    def __init__(self, device: str):
        import torch
        from huggingface_hub import hf_hub_download
        from .enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints

        from huggingface_hub import try_to_load_from_cache

        def get(spec):
            repo, name, rev = spec
            hit = try_to_load_from_cache(repo, name, revision=rev)  # local only: no internet needed once downloaded
            if isinstance(hit, str):
                return Path(hit)
            return Path(hf_hub_download(repo_id=repo, filename=name, revision=rev))

        ckpts = ESRGANUpscalerCheckpoints(
            **{k: get(v) for k, v in _FILES.items()},
            negative_embedding_key="string_to_param.*",
            loras={k: get(v) for k, v in _LORAS.items()},
        )
        self.torch = torch
        # float16 on the Mac GPU: bfloat16 can't be turned back into a picture by numpy
        # ("Got unsupported ScalarType BFloat16"), and older Apple GPUs run float16 faster anyway.
        dtype = torch.float16 if device == "mps" else torch.float32
        _patch_tensor_to_image()
        self.device = torch.device(device)
        self.profile = memory_profile()
        self.upscaler = _make_upscaler_class(ESRGANUpscaler)(checkpoints=ckpts, device=self.device, dtype=dtype)
        self.upscaler.esrgan_tile = self.profile["esrgan_tile"]
        self.upscaler.esrgan_overlap = self.profile["esrgan_overlap"]
        # Speed: fold the two detail LoRAs into the model's weights once, instead of running them as extra
        # layers in every step of every tile (they stay at the enhancer's own strengths).
        self.merged = 0
        try:
            self.merged = _merge_unet_loras(self.upscaler.manager, {
                "more_details": DEFAULTS["lora_more_details"], "sdxl_render": DEFAULTS["lora_sdxl_render"]})
        except Exception as e:  # still works, just slower
            print("[enhancer] couldn't merge the detail LoRAs:", e, flush=True)
        # Memory: compute attention in chunks of 1024 query tokens on smaller Macs (ProtoArt Beautify's
        # setting). Cuts attention's working memory ~4x, which keeps an 8 GB Mac out of swap.
        self.attention = []
        if self.profile["slice"]:
            try:
                self.attention = _slice_attention(self.profile["slice"], self.upscaler.sd.unet,
                                                  self.upscaler.sd.lda, self.upscaler.controlnet)
            except Exception as e:
                print("[enhancer] couldn't turn on sliced attention:", e, flush=True)
        self.sliced = len(self.attention)
        _free_memory(self.device)
        self.turbo = None        # SwitchableLora, loaded the first time Turbo is used
        self.turbo_error = ""

    def enhance(self, pil_image, seed, cancel_event=None, on_progress=None, on_preview=None, **opts):
        from refiners.foundationals.latent_diffusion import solvers

        o = dict(DEFAULTS)
        o.update({k: v for k, v in opts.items() if v is not None and k in DEFAULTS})
        side = work_side(o)
        o["work_size"] = side
        if o.get("instant"):
            return self._instant(pil_image, o, side, cancel_event, on_progress)
        # Tiles: the fewest that fit this Mac's per-tile budget (a 768 px picture is one tile on an 8 GB Mac,
        # instead of many overlapping 512 px ones; the attention memory per tile stays bounded). The picture is
        # padded to a multiple of 64 px for that and cropped back afterwards. (ProtoArt Beautify: 182 s -> 92 s.)
        ow, oh = output_size(pil_image.width, pil_image.height, float(o["upscale_factor"]), side)
        lat_h, lat_w = -(-oh // 64) * 8, -(-ow // 64) * 8
        o["tile_height"], o["tile_width"] = choose_tile(lat_h, lat_w, self.profile["tile_tokens"])
        # Low memory: hand cached GPU memory back to macOS after every pass, and smaller attention chunks.
        low = bool(self.profile["low_memory"])
        self.upscaler.low_memory = low
        for layer in self.attention:
            layer.slice_size = self.profile["slice"] // 2 if low else self.profile["slice"]
        solver = o["solver"] if o["solver"] in SOLVERS else "DDIM"
        use_turbo = bool(o.get("turbo"))
        if use_turbo and self.turbo is None:
            self.turbo = self._load_turbo()  # may download ~135 MB the first time (raises if offline)
        if self.turbo is not None:
            self.turbo.set(use_turbo)
        # Turbo needs no negative-prompt pass: half the work per step
        self.upscaler.sd.classifier_free_guidance = not use_turbo
        if use_turbo:
            solver = "LCMSolver"
            o["num_inference_steps"] = max(4, min(int(o["num_inference_steps"]), TURBO_MAX_STEPS))
            # with an LCM, the guidance scale isn't used (and high values burn the picture)
            o["condition_scale"] = 1.0
        # Stop + progress: refiners has no step callback, so watch every call into the diffusion UNet.
        # Each call is one denoising step on one tile; raising here stops the enhancer within seconds.
        total = _estimate_unet_calls((lat_w, lat_h), o)
        state = {"calls": 0}

        def hook(_module, _inputs):
            if cancel_event is not None and cancel_event.is_set():
                raise EnhanceCancelled()
            state["calls"] += 1
            if on_progress:
                on_progress(min(state["calls"], total), max(total, state["calls"]))

        self.upscaler.on_preview = on_preview  # live preview: the latest denoised tile (the whole picture if one tile)
        handle = None
        unet = _find_unet(self.upscaler)
        if unet is not None:
            handle = unet.register_forward_pre_hook(hook)
        try:
            with self.torch.no_grad():
                result = self._run(pil_image, o, solver, seed, solvers)
                img = result[0]
                box = getattr(self.upscaler, "crop_box", None)
                if box and (box[2], box[3]) != img.size:
                    img = img.crop(box)  # the padding added for the tile planner
                return img, dict(result[1], solver=("Turbo (LCM)" if use_turbo else solver))
        except EnhanceCancelled:
            raise
        except Exception:
            # refiners may wrap our exception in its own error type; a set cancel flag means it was a Stop
            if cancel_event is not None and cancel_event.is_set():
                raise EnhanceCancelled()
            raise
        finally:
            if handle is not None:
                handle.remove()
            _free_memory(self.device)

    def _instant(self, pil_image, o, side, cancel_event, on_progress):
        """Instant: the 4x-UltraSharp upscaler alone, then resized to the target. Seconds instead of minutes,
        but it only sharpens what's there (no new detail)."""
        from PIL import Image
        if cancel_event is not None and cancel_event.is_set():
            raise EnhanceCancelled()
        img = pil_image.convert("RGB")
        w, h = output_size(img.width, img.height, float(o["upscale_factor"]), side)
        # upscale only what's needed: shrink first if the picture is bigger than a 4x pass needs
        if img.width * 4 > w * 1.5:
            img = img.resize((max(8, -(-w // 4)), max(8, -(-h // 4))), Image.Resampling.LANCZOS)
        if on_progress:
            on_progress(0, 1)
        try:
            with self.torch.no_grad():
                big = self.upscaler.esrgan.upscale_with_tiling(img, tile=self.upscaler.esrgan_tile,
                                                               overlap=self.upscaler.esrgan_overlap)
        finally:
            _free_memory(self.device)
        if on_progress:
            on_progress(1, 1)
        if big.size != (w, h):
            big = big.resize((w, h), Image.Resampling.LANCZOS)
        return big, dict(o, solver="Instant (upscaler only)", num_inference_steps=0)

    def _load_turbo(self):
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from refiners.fluxion.utils import load_from_safetensors
        hit = try_to_load_from_cache(*TURBO_LORA)
        path = hit if isinstance(hit, str) else hf_hub_download(*TURBO_LORA)  # offline -> retried online by the app
        tensors = load_from_safetensors(path)
        # The tile ControlNet is plugged into the UNet and has look-alike attention layers ("attached twice"
        # error): unplug it so the LoRA can only match the UNet's own layers, then plug it back in.
        cn = getattr(self.upscaler, "controlnet", None)
        if cn is not None:
            cn.eject()
        try:
            return SwitchableLora(self.upscaler.manager, "lcm", tensors, TURBO_SCALE)
        finally:
            if cn is not None:
                cn.inject()

    def _run(self, pil_image, o, solver, seed, solvers):
        with self.torch.no_grad():
            return self.upscaler.upscale(
                image=pil_image.convert("RGB"),
                prompt=o["prompt"], negative_prompt=o["negative_prompt"],
                upscale_factor=float(o["upscale_factor"]),
                controlnet_scale=float(o["controlnet_scale"]), controlnet_scale_decay=float(o["controlnet_decay"]),
                condition_scale=float(o["condition_scale"]),
                tile_size=(int(o["tile_height"]), int(o["tile_width"])),
                denoise_strength=float(o["denoise_strength"]),
                num_inference_steps=int(o["num_inference_steps"]),
                # (merged into the weights at load when possible; otherwise set per run as before)
                loras_scale=None if self.merged else {"more_details": float(o["lora_more_details"]),
                                                      "sdxl_render": float(o["lora_sdxl_render"])},
                solver_type=_solver_type(solvers, solver, self.device),
                downscale_size=int(o["work_size"]),
                # decode/encode the big upscaled picture in 512 px pieces on small Macs (1024 by default):
                # the whole-image pass is what pushed an 8 GB Mac into swapping and freezing
                autoencoder_tile_length=self.profile["vae_tile"],
                seed=int(seed),
            ), o


def _patch_tensor_to_image():
    """Make refiners' tensor -> PIL conversion work for any float type (casts to float32 first)."""
    import sys
    try:
        import refiners.fluxion.utils as fu
    except Exception:
        return
    original = getattr(fu, "tensor_to_image", None)
    if original is None or getattr(original, "_protoart_safe", False):
        return

    def safe(tensor, *a, **kw):
        return original(tensor.float() if hasattr(tensor, "float") else tensor, *a, **kw)

    safe._protoart_safe = True
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "tensor_to_image", None) is original:
                setattr(mod, "tensor_to_image", safe)
        except Exception:
            pass


def _ram_gb():
    import os
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0


class EnhanceCancelled(Exception):
    pass


def _find_unet(upscaler):
    """The SD UNet inside refiners' upscaler (attribute names differ between versions)."""
    sd = getattr(upscaler, "sd", None)
    for obj in (sd, upscaler):
        u = getattr(obj, "unet", None) if obj is not None else None
        if u is not None and hasattr(u, "register_forward_pre_hook"):
            return u
    return None


def _count_tiles(lat_h, lat_w, th, tw, overlap=8):
    """How many tiles refiners will make (its own tiling function when available)."""
    try:
        from refiners.foundationals.latent_diffusion.multi_diffusion import MultiDiffusion, Size
        return len(MultiDiffusion.generate_latent_tiles(size=Size(height=lat_h, width=lat_w),
                                                         tile_size=Size(height=th, width=tw), min_overlap=overlap))
    except Exception:
        pass
    import math
    nx = 1 if tw >= lat_w else math.ceil((lat_w - overlap) / max(1, tw - overlap))
    ny = 1 if th >= lat_h else math.ceil((lat_h - overlap) / max(1, th - overlap))
    return nx * ny


def choose_tile(lat_h, lat_w, budget):
    """Fewest / cheapest diffusion tiles (latent units, multiples of 8) that fit the per-tile token budget.
    Tiles stay at least 64 latent (512 px) where the picture allows: SD 1.5 loses coherence below that.
    (ProtoArt Beautify's planner.)"""
    if lat_h * lat_w <= budget:
        return lat_h, lat_w
    best = None
    for th in range(min(64, lat_h), lat_h + 1, 8):
        for tw in range(min(64, lat_w), lat_w + 1, 8):
            tokens = th * tw
            if tokens > budget:
                continue
            n = _count_tiles(lat_h, lat_w, th, tw)
            cost = n * (tokens * (1 + tokens / 12000) + 1500)  # convs ~ tokens, attention ~ tokens^2, + per tile
            if best is None or cost < best[0]:
                best = (cost, th, tw)
    return (best[1], best[2]) if best else (64, 64)


def _estimate_unet_calls(latent_wh, o):
    """Number of UNet calls: tiles x denoising steps (for the progress bar / ETA)."""
    lat_w, lat_h = latent_wh
    n = _count_tiles(lat_h, lat_w, int(o["tile_height"]), int(o["tile_width"]))
    steps = max(1, int(o["num_inference_steps"]) - int(int(o["num_inference_steps"]) * (1 - float(o["denoise_strength"]))))
    return n * steps


def _free_memory(device):
    import gc
    gc.collect()
    try:
        import torch
        if getattr(device, "type", str(device)) == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _pad_to_multiple(image, m):
    import numpy as np
    from PIL import Image
    w, h = image.size
    pw, ph = -(-w // m) * m, -(-h // m) * m
    if (pw, ph) == (w, h):
        return image
    arr = np.asarray(image.convert("RGB"))
    mode = "reflect" if (ph - h) < h and (pw - w) < w else "edge"
    return Image.fromarray(np.pad(arr, ((0, ph - h), (0, pw - w), (0, 0)), mode=mode))


def _make_upscaler_class(base):
    """The ESRGAN upscaler with ProtoArt Beautify's memory tuning: ESRGAN in small tiles, the picture padded to
    a multiple of 64 px for the tile planner, and the GPU cache handed back to macOS after every pass."""
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import MultiUpscaler

    class ProtoartUpscaler(base):
        esrgan_tile = 512
        esrgan_overlap = 64
        low_memory = False
        crop_box = None
        on_preview = None

        def pre_upscale(self, image, upscale_factor, **_):
            image = self.esrgan.upscale_with_tiling(image, tile=self.esrgan_tile, overlap=self.esrgan_overlap)
            _free_memory(self.device)
            image = MultiUpscaler.pre_upscale(self, image=image, upscale_factor=upscale_factor / 4)
            self.crop_box = (0, 0, image.width, image.height)
            return _pad_to_multiple(image, 64)

        def diffuse_target(self, x, step, target):
            out = super().diffuse_target(x=x, step=step, target=target)
            if self.on_preview is not None:
                try:
                    self.on_preview(out)
                except Exception:
                    self.on_preview = None
            if self.low_memory and getattr(self.device, "type", "") == "mps":
                import torch
                torch.mps.empty_cache()
            return out

    return ProtoartUpscaler


def _cpu_first(cls):
    """refiners' DPM and LCM solvers compute their noise schedule in float64, which Apple GPUs (MPS) can't
    hold: build them on the CPU, then move them to the GPU in the working precision."""
    class AppleSafe(cls):
        def __init__(self, *a, device="cpu", dtype=None, **kw):
            import torch
            super().__init__(*a, device="cpu", dtype=dtype or torch.float32, **kw)
            if str(device) != "cpu":
                self.to(device=device)
    AppleSafe.__name__ = "Apple" + cls.__name__
    return AppleSafe


def _solver_type(solvers, name, device=None):
    """refiners' Euler solver defaults to fractional timesteps, which crash its add-noise step
    ("tensors used as indices must be long..."); give it whole-number timesteps instead."""
    cls = getattr(solvers, name)
    if name in ("DPMSolver", "LCMSolver") and device is not None and getattr(device, "type", str(device)) == "mps":
        return _cpu_first(cls)
    if name != "Euler":
        return cls
    import dataclasses
    import functools
    params = dataclasses.replace(cls.default_params, timesteps_spacing=solvers.TimestepSpacing.LINSPACE_ROUNDED)
    return functools.partial(cls, params=params)


class SwitchableLora:
    """A LoRA kept as its small low-rank factors and merged into / taken out of the UNet's weights on demand:
    no cost per step, and it can be switched on and off without reloading the model.
    (Same approach as ProtoArt Beautify's enhancer.)"""

    def __init__(self, manager, name, tensors, scale):
        from refiners.fluxion.adapters.lora import LoraAdapter
        expected = sum(1 for k in tensors if k.endswith(".weight")) // 2
        manager.add_loras(name, tensors=tensors, scale=scale)
        self.parts = []
        for ad in [m for m in manager.unet.modules() if isinstance(m, LoraAdapter)]:
            lora = ad.loras.get(name)
            if lora is None:
                continue
            self.parts.append((ad.target.weight, lora.up.weight.detach().clone(),
                               lora.down.weight.detach().clone(), float(lora.scale)))
            ad.remove_lora(name)
            if not list(ad.lora_layers):
                ad.eject()
        if expected and len(self.parts) < expected * 0.9:
            raise RuntimeError(f"only {len(self.parts)} of {expected} Turbo layers attached")
        self.on = False

    def _apply(self, sign):
        import torch
        with torch.no_grad():
            for weight, up, down, s in self.parts:
                u, d = up.float(), down.float()
                delta = u @ d if weight.ndim == 2 else (u.flatten(1) @ d.flatten(1)).reshape(weight.shape)
                weight.add_((sign * s * delta).to(device=weight.device, dtype=weight.dtype))

    def set(self, on):
        if on != self.on:
            self._apply(1.0 if on else -1.0)
            self.on = on


def _merge_unet_loras(manager, scales):
    """Fold LoRAs that were loaded as extra layers into the UNet's own weights (at the given strengths) and
    remove the extra layers. Their text-encoder part stays as it is (it runs once per picture), at the same
    strength. Returns how many layers were merged."""
    import torch
    from refiners.fluxion.adapters.lora import LoraAdapter
    if manager is None:
        return 0
    try:
        manager.update_scales(scales)
    except Exception:
        pass
    n = 0
    with torch.no_grad():
        for ad in [m for m in manager.unet.modules() if isinstance(m, LoraAdapter)]:
            for name in list(ad.loras):
                if name not in scales:
                    continue
                lora = ad.loras[name]
                weight = ad.target.weight
                u, d = lora.up.weight.float(), lora.down.weight.float()
                delta = u @ d if weight.ndim == 2 else (u.flatten(1) @ d.flatten(1)).reshape(weight.shape)
                weight.add_((float(lora.scale) * delta).to(device=weight.device, dtype=weight.dtype))
                ad.remove_lora(name)
                n += 1
            if not list(ad.lora_layers):
                ad.eject()
    return n


def _slice_attention(slice_size, *models):
    """Turn on refiners' built-in sliced attention for every attention layer of the given models."""
    from refiners.fluxion.layers.attentions import ScaledDotProductAttention
    seen, layers = set(), []
    for model in models:
        for m in model.modules():
            if isinstance(m, ScaledDotProductAttention) and id(m) not in seen:
                m.slice_size = slice_size
                seen.add(id(m))
                layers.append(m)
    return layers
