"""💡 Relight with IC-Light (lllyasviel/IC-Light, Apache-2.0 code): the subject is cut out and drawn again under new
light described in words, on a matching new background. It runs on one of your Stable Diffusion 1.5 models: IC-Light's
weights (iclight_sd15_fc, ~1.7 GB, downloaded once from Hugging Face) are added on top of that model's UNet, whose first
layer takes the subject's picture as 4 extra channels. Our own implementation of the method in IC-Light's demo
(gradio_demo.py): a gradient start picture for the light's direction, then a second, larger pass.
"""
import numpy as np

REPO = "lllyasviel/ic-light"
FILE = "iclight_sd15_fc.safetensors"
FILE_BG = "iclight_sd15_fbc.safetensors"  # lit like a reference picture (IC-Light's background-conditioned model)
MB = 1720
DIRECTIONS = ("none", "left", "right", "top", "bottom")
ADDED = "best quality"
NEGATIVE = "lowres, bad anatomy, bad hands, cropped, worst quality"


def downloaded(file=FILE):
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, file, local_files_only=True)
        return True
    except Exception:
        return False


def weights_path(download=False, file=FILE):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, file, local_files_only=not download)


def patch_unet(unet, offset_file):
    """In place: a wider first layer (8 channels for the words model, 12 for the reference-picture one; the new ones
    start at zero), IC-Light's offsets added to every weight, and a forward that appends the subject's latent (and the
    reference's) from cross_attention_kwargs["concat_conds"] to the noisy latent."""
    import torch
    from safetensors import safe_open
    with safe_open(offset_file, framework="pt") as f:
        channels = int(f.get_slice("conv_in.weight").get_shape()[1]) if "conv_in.weight" in f.keys() else 8
    with torch.no_grad():
        old = unet.conv_in
        new = torch.nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding).to(old.weight.device, old.weight.dtype)
        new.weight.zero_()
        new.weight[:, :4].copy_(old.weight)
        new.bias = old.bias
        unet.conv_in = new
        params = dict(unet.named_parameters())
        params.update(dict(unet.named_buffers()))
        with safe_open(offset_file, framework="pt") as f:  # one tensor at a time: no second copy of the UNet in memory
            missing = [k for k in params if k not in f.keys()]
            if missing:
                raise ValueError(f"IC-Light's weights don't fit this model ({len(missing)} parts missing)")
            for k in f.keys():
                p = params.get(k)
                if p is None:
                    continue
                off = f.get_tensor(k)
                p.data.copy_((p.data.float() + off.to(p.device).float()).to(p.dtype))
    original = unet.forward

    def forward(sample, timestep, encoder_hidden_states, **kw):
        cak = dict(kw.get("cross_attention_kwargs") or {})
        c = cak.pop("concat_conds").to(sample)
        c = torch.cat([c] * (sample.shape[0] // c.shape[0]), dim=0)
        kw["cross_attention_kwargs"] = cak or None
        return original(torch.cat([sample, c], dim=1), timestep, encoder_hidden_states, **kw)

    unet.forward = forward
    return unet


def foreground(image, alpha):
    """The subject on mid grey (127), as IC-Light expects: PIL RGB + matte (h, w) 0..1 -> uint8 array."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    a = np.asarray(alpha, dtype=np.float32)[..., None]
    return (127 + (rgb - 127) * a).clip(0, 255).astype(np.uint8)


def gradient(direction, w, h):
    """The light's direction as a start picture: bright on the side the light comes from."""
    if direction in ("left", "right"):
        g = np.linspace(255, 0, w) if direction == "left" else np.linspace(0, 255, w)
        img = np.tile(g, (h, 1))
    else:
        g = np.linspace(255, 0, h) if direction == "top" else np.linspace(0, 255, h)
        img = np.tile(g[:, None], (1, w))
    return np.stack((img,) * 3, axis=-1).astype(np.uint8)


def sizes(w, h, long_side):
    """First-pass size (multiples of 64, long side = long_side) for a w x h picture."""
    k = long_side / max(w, h)
    return max(256, int(round(w * k / 64)) * 64), max(256, int(round(h * k / 64)) * 64)


def fit(arr, w, h):
    """Resize and centre-crop an uint8 array to w x h (like IC-Light's resize_and_center_crop)."""
    from PIL import Image
    im = Image.fromarray(arr)
    k = max(w / im.width, h / im.height)
    im = im.resize((int(round(im.width * k)), int(round(im.height * k))), Image.LANCZOS)
    left, top = (im.width - w) / 2, (im.height - h) / 2
    return np.asarray(im.crop((left, top, left + w, top + h)))


def to_tensor(arr):
    import torch
    t = torch.from_numpy(np.stack([arr], axis=0)).float() / 127.0 - 1.0  # so that 127 is exactly 0
    return t.movedim(-1, 1)


def place_back(original, alpha, relit, relight_bg=False):
    """The relit subject (IC-Light's picture, centre-cropped to its own size by fit()) on the original picture's
    background: the original is cropped and scaled the same way, and the subject's matte decides what comes from where
    (with a slightly softened edge). relight_bg: the background keeps everything in it but takes on the new light's
    brightness and colour, read from the background IC-Light drew (a soft light map; no new shadows).
    original: PIL RGB; alpha: (h, w) 0..1; relit: PIL RGB -> PIL RGB, relit's size."""
    from PIL import Image, ImageFilter
    w, h = relit.size
    bg = Image.fromarray(fit(np.asarray(original.convert("RGB")), w, h))
    a = Image.fromarray((np.clip(np.asarray(alpha, dtype=np.float32), 0, 1) * 255).astype(np.uint8))
    a = Image.fromarray(fit(np.stack([np.asarray(a)] * 3, axis=-1), w, h)[..., 0])
    if relight_bg:
        bg = relight_background(bg, relit, a)
    a = a.filter(ImageFilter.GaussianBlur(max(1, round(max(w, h) / 512))))
    return Image.composite(relit, bg, a)


def relight_background(bg, relit, matte):
    """bg (PIL RGB) lit like relit's background: both are blurred a lot with the subject left out (so only the light
    and colour remain, not the shapes), and bg is multiplied by new light / old light, per colour."""
    from PIL import Image, ImageFilter
    w, h = bg.size
    r = max(4, round(max(w, h) / 14))
    keep = 255 - np.asarray(matte.filter(ImageFilter.MaxFilter(9)), dtype=np.float32)  # background only, a bit away from the subject
    m = Image.fromarray(keep.astype(np.uint8))

    def light(img):
        ar = np.asarray(img, dtype=np.float32) * (keep[..., None] / 255.0)
        num = np.stack([np.asarray(Image.fromarray(ar[..., c].clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)),
                                   dtype=np.float32) for c in range(3)], axis=-1)
        den = np.asarray(m.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)[..., None]
        return num / np.maximum(den / 255.0, 1e-3), den  # the average colour of the background nearby

    new, den = light(relit)
    old, _ = light(bg)
    gain = np.clip((new + 2.0) / (old + 2.0), 0.2, 3.0)
    weak = (den < 8)  # hardly any background around (a subject filling the picture): leave it as it was
    gain[weak[..., 0]] = 1.0
    out = np.asarray(bg, dtype=np.float32) * gain
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ------------------------------------------------------------------ finishing at full size (Keep mine, relit…)
def crop_size(size, relit_size):
    """The original's size scaled so that its relit part is exactly relit_size (for results not kept at full size)."""
    W, H = size
    w, h = relit_size
    k = max(w / W, h / H)
    return max(1, round(W * k)), max(1, round(H * k))


def crop_box(W, H, w, h):
    """The part of a W x H picture that fit() keeps for a w x h result: (left, top, right, bottom) in the picture."""
    k = max(w / W, h / H)
    cw, ch = w / k, h / k
    left, top = (W - cw) / 2, (H - ch) / 2
    return int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch))


DIRS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}


def light_direction(relit, matte, direction="none"):
    """Where the light comes from, as (dx, dy) on the picture (unit length; y down): the direction you picked, or the
    bright side of the background IC-Light drew."""
    if direction in DIRS:
        return DIRS[direction]
    from PIL import ImageFilter
    small = relit.convert("L").resize((64, 64))
    m = np.asarray(matte.resize((64, 64)), dtype=np.float32) / 255.0
    lum = np.asarray(small.filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) * (1 - m)
    lum = np.clip(lum - np.percentile(lum[m < 0.5] if (m < 0.5).any() else lum, 40), 0, None)
    if lum.sum() < 1:
        return (0.0, -1.0)
    yy, xx = np.mgrid[0:64, 0:64]
    dx, dy = (lum * (xx - 31.5)).sum() / lum.sum(), (lum * (yy - 31.5)).sum() / lum.sum()
    n = (dx * dx + dy * dy) ** 0.5
    return (dx / n, dy / n) if n > 1.5 else (0.0, -1.0)


def depth_shading(depth, light, strength=0.6):
    """How much each part of the scene faces the light, from a depth map (PIL L, near = bright): surfaces turned
    towards the light brighten and those turned away darken. -> (h, w) gain around 1."""
    d = np.asarray(depth, dtype=np.float32) / 255.0
    import cv2
    d = cv2.GaussianBlur(d, (0, 0), max(1.5, max(d.shape) / 300))
    gy, gx = np.gradient(d)
    scale = max(d.shape) / 6.0
    nx, ny, nz = -gx * scale, -gy * scale, np.ones_like(d)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    lx, ly = light
    lz = 0.55
    ln = (lx * lx + ly * ly + lz * lz) ** 0.5
    shade = (nx * lx + ny * ly + nz * lz) / (norm * ln)
    # a little from the near/far too: what's nearer the light's side of the picture catches more of it
    H, W = d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    side = ((xx / W - 0.5) * lx + (yy / H - 0.5) * ly) * 0.5
    shade = shade + side
    shade = (shade - shade.mean()) / (shade.std() + 1e-6)
    return np.clip(1.0 + shade * 0.18 * strength, 0.55, 1.6)


def cast_shadow(matte, light, darkness=0.45):
    """A soft shadow of the subject on its background, thrown away from the light, anchored at the subject's feet.
    -> (h, w) multiplier (1 = no shadow)."""
    from PIL import Image, ImageFilter
    a = np.asarray(matte, dtype=np.float32) / 255.0
    H, W = a.shape
    ys, xs = np.nonzero(a > 0.5)
    if len(ys) < 50:
        return np.ones_like(a)
    top, bottom = ys.min(), ys.max()
    hgt = bottom - top + 1
    lx, ly = light
    import cv2
    if abs(ly) < 0.6 or ly > 0:  # light from a side (or from below): the shadow falls along the ground, away from it
        sx = lx * 0.9  # (forward map: the top of the subject lands away from the light)
        M = np.float32([[1, sx * 1.0, -sx * bottom], [0, 0.35, bottom * 0.65]])
        sh = cv2.warpAffine(a, M, (W, H))
    else:  # light from above: a soft pool under the subject
        M = np.float32([[1.1, 0, -0.05 * W], [0, 0.12, bottom * 0.88 + hgt * 0.02]])
        sh = cv2.warpAffine(a, M, (W, H))
    sh = np.asarray(Image.fromarray((sh * 255).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(max(2, hgt / 40)))),
                    dtype=np.float32) / 255.0
    return 1.0 - sh * darkness


def blend_in(subject_rgb, bg_rgb, matte, amount=0.3):
    """The subject's colours nudged towards the light of the background around it (a soft colour match), so the two
    look lit by the same light. Arrays float 0..255 (h, w, 3); matte (h, w) 0..1."""
    import cv2
    m = matte[..., None]
    ring = cv2.GaussianBlur(matte, (0, 0), max(4, max(matte.shape) / 40))
    near = np.clip(ring - matte, 0, 1)[..., None]  # the background just around the subject
    if near.sum() < 10 or (m > 0.5).sum() < 10:
        return subject_rgb
    bg_mean = (bg_rgb * near).sum((0, 1)) / near.sum()
    sub_mean = (subject_rgb * m).sum((0, 1)) / m.sum()
    shift = (bg_mean / (sub_mean + 1.0))
    shift = shift / shift.mean()  # colour only, not brightness
    tinted = subject_rgb * (1 + (shift - 1) * amount)
    return tinted


def warm_and_tint(arr, warmth=0.0, colour=None, amount=0.25):
    """The light's warmth (-1 cold blue … +1 warm orange) and an optional light colour (#rrggbb), on an RGB float array."""
    out = arr
    if warmth:
        out = out * np.array([1 + 0.12 * warmth, 1 + 0.02 * warmth, 1 - 0.12 * warmth], dtype=np.float32)
    if colour:
        c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) + 1
        out = out * (1 + (c / c.mean() - 1) * amount)
    return out


def finish(original, alpha, relit, mode="keep", depth=None, direction="none", shadow=True, blend=True,
           warmth=0.0, colour=None, upscale=None):
    """Relight's result at the original picture's full size.
    original: PIL RGB (full size); alpha: (H, W) 0..1 matte of the subject; relit: IC-Light's picture (w x h, the
    centre part fit() took); mode: keep / keep_relit / new; depth: PIL L depth map of the original (for keep_relit);
    upscale(img, (cw, ch)) -> img: makes the relit picture big again (an upscaler; plain resizing if None)."""
    from PIL import Image, ImageFilter
    W, H = original.size
    w, h = relit.size
    box = crop_box(W, H, w, h)
    cw, ch = box[2] - box[0], box[3] - box[1]
    big = upscale(relit, (cw, ch)) if upscale else relit.resize((cw, ch), Image.LANCZOS)
    if big.size != (cw, ch):
        big = big.resize((cw, ch), Image.LANCZOS)
    full = np.asarray(original.convert("RGB"), dtype=np.float32).copy()
    region = full[box[1]:box[3], box[0]:box[2]]
    rel = np.asarray(big, dtype=np.float32)
    a8 = Image.fromarray((np.clip(np.asarray(alpha, dtype=np.float32), 0, 1) * 255).astype(np.uint8))
    if a8.size != (W, H):
        a8 = a8.resize((W, H), Image.LANCZOS)
    a_full = a8.crop(box)
    if mode == "new":
        out = rel
    else:
        bg = region
        light = light_direction(relit, a8.crop(box).resize((w, h)), direction)
        if mode == "keep_relit":
            small_bg = Image.fromarray(region.clip(0, 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
            lit_small = relight_background(small_bg, relit, a_full.resize((w, h), Image.LANCZOS))
            gain = (np.asarray(lit_small, dtype=np.float32) + 2) / (np.asarray(small_bg, dtype=np.float32) + 2)
            gain = np.asarray(Image.fromarray(np.clip(gain * 85, 0, 255).astype(np.uint8)).resize((cw, ch), Image.BICUBIC),
                              dtype=np.float32) / 85.0  # a smooth light map, made big again
            if depth is not None:
                d = depth.crop(box) if depth.size == (W, H) else depth.resize((W, H)).crop(box)
                gain = gain * depth_shading(d, light)[..., None]
            bg = region * gain
        if shadow:
            bg = bg * cast_shadow(a_full, light)[..., None]
        m = np.asarray(a_full.filter(ImageFilter.GaussianBlur(max(1, round(max(cw, ch) / 700)))), dtype=np.float32) / 255.0
        subj = blend_in(rel, bg, m) if blend else rel
        out = subj * m[..., None] + bg * (1 - m[..., None])
    out = warm_and_tint(out, warmth, colour)
    full[box[1]:box[3], box[0]:box[2]] = out
    return Image.fromarray(full.clip(0, 255).astype(np.uint8))
