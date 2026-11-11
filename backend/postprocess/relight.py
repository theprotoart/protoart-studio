"""💡 Relight with IC-Light (lllyasviel/IC-Light, Apache-2.0 code): the subject is cut out and drawn again under new
light described in words, on a matching new background. It runs on one of your Stable Diffusion 1.5 models: IC-Light's
weights (iclight_sd15_fc, ~1.7 GB, downloaded once from Hugging Face) are added on top of that model's UNet, whose first
layer takes the subject's picture as 4 extra channels. Our own implementation of the method in IC-Light's demo
(gradio_demo.py): a gradient start picture for the light's direction, then a second, larger pass.
"""
import numpy as np

REPO = "lllyasviel/ic-light"
FILE = "iclight_sd15_fc.safetensors"
MB = 1720
DIRECTIONS = ("none", "left", "right", "top", "bottom")
ADDED = "best quality"
NEGATIVE = "lowres, bad anatomy, bad hands, cropped, worst quality"


def downloaded():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, FILE, local_files_only=True)
        return True
    except Exception:
        return False


def weights_path(download=False):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, FILE, local_files_only=not download)


def patch_unet(unet, offset_file):
    """In place: an 8-channel first layer (the new 4 start at zero), IC-Light's offsets added to every weight, and a
    forward that appends the subject's latent (cross_attention_kwargs["concat_conds"]) to the noisy latent."""
    import torch
    from safetensors import safe_open
    with torch.no_grad():
        old = unet.conv_in
        new = torch.nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding).to(old.weight.device, old.weight.dtype)
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
