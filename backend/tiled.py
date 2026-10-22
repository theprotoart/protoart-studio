"""Tiled diffusion upscale (MultiDiffusion, Bar-Tal et al. 2023), for SD 1.5 and SDXL.

The picture is enlarged, then lightly redrawn by the model in overlapping tiles. At every step each tile's noise
prediction is worked out separately and the predictions are averaged where tiles overlap (with soft weights), so
the whole big picture is denoised as one: no seams, and memory never needs more than one tile at a time. Adds
real detail at sizes the model couldn't draw in one go on a small Mac.
"""
import math

import numpy as np


def _starts(size, tile, stride):
    if size <= tile:
        return [0]
    n = math.ceil((size - tile) / stride) + 1
    return [min(i * stride, size - tile) for i in range(n)]


def _weight(h, w, device, dtype):
    """Soft tile weights (highest in the middle) so overlaps blend smoothly."""
    import torch
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    g = torch.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * 0.5 ** 2))
    return (g / g.max()).to(dtype).clamp(min=1e-3)


def upscale(pipe, family, image, prompt, negative, scale=2.0, strength=0.35, steps=20, cfg=6.0, seed=0,
            tile=None, overlap=None, on_step=None, emit=None):
    """-> PIL image, `scale` × bigger, redrawn tile by tile."""
    import torch
    from PIL import Image
    emit = emit or (lambda m: None)
    device = pipe.unet.device
    dtype = pipe.unet.dtype
    xl = family == "sdxl"
    W = int(image.width * scale) // 8 * 8
    H = int(image.height * scale) // 8 * 8
    big = image.convert("RGB").resize((W, H), Image.LANCZOS)
    tile = tile or (1024 if xl else 512)
    overlap = overlap or (tile // 4)
    lt, lo = tile // 8, overlap // 8
    stride = max(8, lt - lo)

    # text
    if xl:
        pe, npe, pooled, npooled = pipe.encode_prompt(prompt, device=device, num_images_per_prompt=1,
                                                      do_classifier_free_guidance=cfg > 1, negative_prompt=negative)
    else:
        pe, npe = pipe.encode_prompt(prompt, device, 1, cfg > 1, negative)
        pooled = npooled = None

    # the enlarged picture as latents (the VAE works in tiles too)
    for name in ("enable_tiling", "enable_vae_tiling"):
        if hasattr(pipe.vae, name):
            getattr(pipe.vae, name)()
            break
    arr = torch.from_numpy(np.asarray(big).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    upcast = pipe.vae.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False)
    if upcast:  # SDXL's own VAE overflows in half precision: it works in full precision (it's small)
        pipe.vae.to(torch.float32)
    vae_dtype = pipe.vae.dtype
    with torch.no_grad():
        lat = pipe.vae.encode(arr.to(device, vae_dtype)).latent_dist.mean * pipe.vae.config.scaling_factor
    lat = lat.to(dtype)

    sched = pipe.scheduler
    sched.set_timesteps(steps, device=device)
    start = min(int(steps * strength), steps)
    timesteps = sched.timesteps[steps - start:]
    if hasattr(sched, "set_begin_index"):
        sched.set_begin_index(steps - start)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32).to(device, dtype)
    latents = sched.add_noise(lat, noise, timesteps[:1]) if len(timesteps) else lat

    lh, lw = latents.shape[-2:]
    ys, xs = _starts(lh, lt, stride), _starts(lw, lt, stride)
    tiles = [(y, x, min(lt, lh), min(lt, lw)) for y in ys for x in xs]
    emit({"type": "status", "msg": f"Tiled upscale: {W} × {H}, {len(tiles)} tiles × {len(timesteps)} steps…"})
    weights = {}
    for i, t in enumerate(timesteps):
        acc = torch.zeros_like(latents, dtype=torch.float32)
        cnt = torch.zeros((1, 1, lh, lw), device=device, dtype=torch.float32)
        for (y, x, th, tw) in tiles:
            chunk = latents[:, :, y:y + th, x:x + tw]
            inp = torch.cat([chunk] * 2) if cfg > 1 else chunk
            inp = sched.scale_model_input(inp, t)
            kw = {}
            if xl:
                ids = torch.tensor([[th * 8, tw * 8, y * 8, x * 8, th * 8, tw * 8]], device=device, dtype=dtype)
                kw["added_cond_kwargs"] = {"text_embeds": torch.cat([npooled, pooled]) if cfg > 1 else pooled,
                                           "time_ids": torch.cat([ids, ids]) if cfg > 1 else ids}
            with torch.no_grad():
                eps = pipe.unet(inp, t, encoder_hidden_states=torch.cat([npe, pe]) if cfg > 1 else pe, **kw).sample
            if cfg > 1:
                u, c = eps.chunk(2)
                eps = u + cfg * (c - u)
            key = (th, tw)
            if key not in weights:
                weights[key] = _weight(th, tw, device, torch.float32)
            wgt = weights[key]
            acc[:, :, y:y + th, x:x + tw] += eps.float() * wgt
            cnt[:, :, y:y + th, x:x + tw] += wgt
        eps_full = (acc / cnt).to(dtype)
        latents = sched.step(eps_full, t, latents).prev_sample
        if on_step:
            on_step(i, len(timesteps))
        if device.type == "mps":
            torch.mps.empty_cache()

    with torch.no_grad():
        dec = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(vae_dtype)).sample
    if upcast:
        pipe.vae.to(torch.float16)
    out = ((dec[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(out)
