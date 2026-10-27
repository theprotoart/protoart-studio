"""📄 Paper cut-out: the subject cut out of the picture like a paper sticker: a torn paper border around it, a
soft shadow under it, printed as a photo, a poster or in black and white, on a see-through, paper or plain
background. The subject is found by ✂ Cut out's model (postprocess/cutout.py); everything else here is drawing
with NumPy / OpenCV (no other model, about a second).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

EDGE_COLOURS = {"white": (250, 248, 242), "cream": (243, 233, 210), "kraft": (196, 160, 112), "black": (24, 24, 24)}
BG_PAPER = (232, 224, 205)


def _noise(h, w, scales=(0.004, 0.012, 0.035, 0.09), seed=0):
    """smooth random wobble in -1..1: a few layers of blurry noise, big to small (fractal noise)"""
    import cv2
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for s in scales:
        gw, gh = max(2, int(w * s)), max(2, int(h * s))
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    out /= tot
    return np.clip(out / (np.abs(out).max() + 1e-6), -1, 1)


def _fibres(h, w, seed=1):
    """paper grain: fine speckle plus short streaks"""
    import cv2
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(n, (0, 0), 0.7)
    streak = cv2.blur(n, (9, 1)) + cv2.blur(n, (1, 7))
    t = 0.6 * fine / (fine.std() + 1e-6) + 0.4 * streak / (streak.std() + 1e-6)
    return np.clip(t / 3, -1, 1)


def _poster(rgb, colours=7):
    import cv2
    small = cv2.bilateralFilter(rgb, 9, 60, 7)
    small = cv2.bilateralFilter(small, 9, 60, 7)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    sample = lab[np.random.default_rng(0).choice(len(lab), min(len(lab), 40000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(sample, colours, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = ((lab[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    lab_q = centres[d.argmin(1)].reshape(rgb.shape).astype(np.uint8)
    out = cv2.cvtColor(lab_q, cv2.COLOR_LAB2RGB)
    return cv2.medianBlur(out, 5)


def _mono(rgb):
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return np.dstack([g, g, g])


def _colour(c, default):
    if isinstance(c, str) and c in EDGE_COLOURS:
        return EDGE_COLOURS[c]
    s = str(c or "").lstrip("#")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def make(image, alpha, edge=2.0, edge_colour="white", rough=0.5, shadow=0.6, print_as="photo", background="transparent", seed=7):
    """image: PIL; alpha: float (h, w) 0..1 (the subject). edge: the border's width in % of the picture's shorter side.
    -> PIL RGBA (transparent background) or RGB."""
    import cv2
    rgb = np.asarray(image.convert("RGB"))
    h, w = alpha.shape
    side = min(h, w)
    solid = (alpha > 0.5).astype(np.uint8)
    if not solid.any():
        raise ValueError("No subject was found to cut out.")
    # --- the border: everything within `edge` of the subject, its outline wobbling like torn paper
    width = max(1.0, float(edge) / 100.0 * side)
    dist = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 5)
    rough = float(np.clip(rough, 0, 1))
    wob = _noise(h, w, seed=seed) * 0.45 + _noise(h, w, scales=(0.05, 0.15, 0.4), seed=seed + 1) * 0.25
    reach = width * (1 + rough * wob) + rough * 1.2 * _noise(h, w, scales=(0.3, 0.7), seed=seed + 2)
    sdf = reach - dist                                    # > 0 inside the sticker
    sticker = np.clip(sdf + 0.5, 0, 1).astype(np.float32)  # 1 px anti-aliased edge
    sticker = np.maximum(sticker, alpha)
    # the torn edge's lighter, fibrous rim (the paper's core shows where it tore)
    rim = np.clip(1 - np.abs(sdf - 1.2) / 1.6, 0, 1) * rough * (0.6 + 0.4 * (_fibres(h, w, seed + 3) > 0.1))
    # --- the paper
    fib = _fibres(h, w, seed + 4)
    ec = np.array(_colour(edge_colour, EDGE_COLOURS["white"]), np.float32)
    paper = ec[None, None, :] * (1 + 0.035 * fib[..., None])
    paper = paper + (255 - paper) * (0.55 * rim[..., None])
    # --- the subject, printed on it
    art = {"poster": _poster, "mono": _mono}.get(print_as, lambda x: x)(rgb).astype(np.float32)
    art = art * (1 + 0.02 * fib[..., None])  # a touch of the paper's grain shows through the print
    a = alpha[..., None]
    front = art * a + paper * (1 - a)
    # --- the shadow under the sticker
    s = float(np.clip(shadow, 0, 1))
    off = max(1, int(round(side * 0.008 * (0.5 + s))))
    blur = max(1.0, side * 0.006 * (0.5 + 2 * s))
    m = np.float32([[1, 0, off], [0, 1, off * 1.4]])
    sh = cv2.warpAffine(sticker, m, (w, h), borderValue=0)
    sh = cv2.GaussianBlur(sh, (0, 0), blur) * 0.75 * s
    # --- onto the background
    if background in (None, "", "transparent"):
        out_a = sticker + sh * (1 - sticker)
        col = (front * sticker[..., None]) / np.maximum(out_a[..., None], 1e-4)  # the shadow itself is black
        rgba = np.dstack([np.clip(col, 0, 255), np.clip(out_a * 255, 0, 255)]).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")
    if background == "paper":
        bg = np.array(BG_PAPER, np.float32)[None, None, :] * (1 + 0.05 * _fibres(h, w, seed + 5)[..., None]
                                                              + 0.04 * _noise(h, w, seed=seed + 6)[..., None])
    elif background == "picture":  # the sticker on its own picture (as if stuck onto the photo)
        bg = rgb.astype(np.float32)
    else:
        bg = np.broadcast_to(np.array(_colour(background, BG_PAPER), np.float32), (h, w, 3))
    bg = bg * (1 - sh[..., None])
    out = front * sticker[..., None] + bg * (1 - sticker[..., None])
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")
