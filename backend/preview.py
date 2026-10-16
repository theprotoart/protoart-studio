"""Live previews while an image is being made (like Draw Things shows).

The model works on a small "latent" picture (1/8 or 1/16 of the final size). A fixed linear mix
of its channels gives a rough colour preview for almost no cost, so no extra model is needed.
Factor tables are the widely used community approximations (as in ComfyUI's latent previews).
Only numpy + Pillow are used, so the MLX engine's separate Python can use this file too.
"""
import base64
import io

import numpy as np

FACTORS = {
    "sd15": ([[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350],
              [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]], [0.0, 0.0, 0.0]),
    "sdxl": ([[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068],
              [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]], [0.1084, -0.0175, -0.0011]),
    # ERNIE-Image uses the FLUX.2 autoencoder: 32 channels (packed as 128 = 32 x 2x2 patches)
    "ernie": ([[0.0058, 0.0113, 0.0073], [0.0495, 0.0443, 0.0836], [-0.0099, 0.0096, 0.0644], [0.2144, 0.3009, 0.3652], [0.0166, -0.0039, -0.0054], [0.0157, 0.0103, -0.016], [-0.0398, 0.0902, -0.0235], [-0.0052, 0.0095, 0.0109], [-0.3527, -0.2712, -0.1666], [-0.0301, -0.0356, -0.018], [-0.0107, 0.0078, 0.0013], [0.0746, 0.009, -0.0941], [0.0156, 0.0169, 0.007], [-0.0034, -0.004, -0.0114], [0.0032, 0.0181, 0.008], [-0.0939, -0.0008, 0.0186], [0.0018, 0.0043, 0.0104], [0.0284, 0.0056, -0.0127], [-0.0024, -0.0022, -0.003], [0.1207, -0.0026, 0.0065], [0.0128, 0.0101, 0.0142], [0.0137, -0.0072, -0.0007], [0.0095, 0.0092, -0.0059], [0.0, -0.0077, -0.0049], [-0.0465, -0.0204, -0.0312], [0.0095, 0.0012, -0.0066], [0.029, -0.0034, 0.0025], [0.022, 0.0169, -0.0048], [-0.0332, -0.0457, -0.0468], [-0.0085, 0.0389, 0.0609], [-0.0076, 0.0003, -0.0043], [-0.0111, -0.046, -0.0614]], [-0.0329, -0.0718, -0.0851]),
    # FLUX.1 autoencoder: 16 channels (community approximation, as in ComfyUI)
    "flux": ([[-0.0346, 0.0244, 0.0681], [0.0034, 0.0210, 0.0687], [0.0275, -0.0668, -0.0433], [-0.0174, 0.0160, 0.0617],
              [0.0859, 0.0721, 0.0329], [0.0004, 0.0383, 0.0115], [0.0405, 0.0861, 0.0915], [-0.0236, -0.0185, -0.0259],
              [-0.0245, 0.0250, 0.1180], [0.1008, 0.0755, -0.0421], [-0.0515, 0.0201, 0.0011], [0.0428, -0.0012, -0.0036],
              [0.0817, 0.0765, 0.0749], [-0.1264, -0.0522, -0.1103], [-0.0280, -0.0881, -0.0499], [-0.1262, -0.0982, -0.0778]],
             [-0.0329, -0.0718, -0.0851]),
}


def unpack_flux_tokens(tokens, height, width):
    """FLUX keeps its latents as a sequence of 2x2 patches: (seq, 64) -> (16, H/8, W/8)."""
    tokens = np.asarray(tokens, dtype=np.float32)
    h2, w2 = height // 16, width // 16
    return tokens.reshape(h2, w2, 16, 2, 2).transpose(2, 0, 3, 1, 4).reshape(16, h2 * 2, w2 * 2)


def _unpack_128(lat):
    c, h, w = lat.shape
    lat = lat.reshape(c // 4, 2, 2, h, w).transpose(0, 3, 1, 4, 2)
    return lat.reshape(c // 4, h * 2, w * 2)


def latent_to_rgb(lat, family):
    """lat: numpy array (C, H, W) for one image. Returns uint8 (H, W, 3)."""
    lat = np.asarray(lat, dtype=np.float32)
    if family == "ernie" and lat.shape[0] == 128:
        lat = _unpack_128(lat)
    factors, bias = FACTORS.get(family) or FACTORS["sd15"]
    f = np.asarray(factors, dtype=np.float32)
    if lat.shape[0] != f.shape[0]:
        raise ValueError("unexpected latent shape for preview")
    s = float(lat.std())
    if family not in ("ernie", "flux") and s > 1.5:
        lat = lat / s  # early, very noisy steps (k-diffusion style samplers scale the noise up)
    rgb = np.einsum("chw,cr->hwr", lat, f) + np.asarray(bias, dtype=np.float32)
    return (np.clip((rgb + 1.0) / 2.0, 0, 1) * 255).astype(np.uint8)


def latent_to_data_url(lat, family, max_side=384, quality=72):
    from PIL import Image
    img = Image.fromarray(latent_to_rgb(lat, family))
    scale = max_side / max(img.size)
    if scale > 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
