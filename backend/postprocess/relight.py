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
