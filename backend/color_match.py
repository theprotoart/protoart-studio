"""Make an edited area sit in the picture: match its light and colour to the original around it.

When a model redraws the whole picture and only the painted part is kept (Edit with words with a painted area),
the redraw often comes out a little lighter or warmer overall, so the kept part shows as a lighter patch with a
halo. This measures the difference between the redraw and the original where the original is kept (outside the
painted area), smoothly, and carries that correction into the painted area before it is pasted back. Fine detail
is left alone: only the slow, broad differences in light and colour are corrected.
"""
import numpy as np
from PIL import Image, ImageFilter


def _blur(a, radius):
    """Gaussian blur of a float array (H, W). PIL's radius is the standard deviation, as here."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma=float(radius), mode="nearest")
    except ImportError:  # 8-bit fallback through PIL (enough for broad light changes)
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-9:
            return a.copy()
        im = Image.fromarray(((a - lo) / (hi - lo) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
        return np.asarray(im, dtype=np.float32) / 255 * (hi - lo) + lo


def _fill_blur(values, known, radius):
    """Blur `values` using only the pixels where `known` > 0 (normalised convolution): holes get the
    surrounding values, spread smoothly."""
    w = _blur(known, radius)
    v = _blur(values * known, radius)
    return v / np.maximum(w, 1e-4)


def match_to_original(result, original, mask, radius=None):
    """result, original: RGB PIL images of the same size; mask: L image, white = the edited area.
    Returns `result` with its broad light and colour inside (and around) the mask matched to `original`."""
    r = np.asarray(result.convert("RGB"), dtype=np.float32)
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    h, w = m.shape
    if radius is None:
        radius = max(8, min(h, w) // 12)
    # where the original is kept and trusted: outside the painted area, a little away from its edge
    grown = np.asarray(mask.convert("L").filter(ImageFilter.MaxFilter(9)), dtype=np.float32) / 255.0
    known = (grown < 0.05).astype(np.float32)
    if known.sum() < 0.02 * h * w:  # almost everything painted: nothing to match against
        return result
    out = r.copy()
    for c in range(3):
        diff = o[..., c] - r[..., c]
        corr = _fill_blur(diff, known, radius)
        out[..., c] = r[..., c] + corr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def keep_tones(enhanced, original, radius=None):
    """The enhanced picture's fine detail with the original's broad colours and light (same size). Used by
    Enhance → Keep the picture's style, so a painting's palette and soft shading don't drift towards a photo."""
    e = np.asarray(enhanced.convert("RGB"), dtype=np.float32)
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    h, w = e.shape[:2]
    if radius is None:
        radius = max(2, min(h, w) // 160)
    out = e.copy()
    for c in range(3):
        out[..., c] = e[..., c] + _blur(o[..., c] - e[..., c], radius)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------------- 🎯 Match colours (Tools)
def _to_lab(rgb):
    """sRGB 0..255 (H, W, 3) -> CIE Lab (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _from_lab(lab):
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883], np.float32)
    m = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]], np.float32)
    c = np.clip(xyz @ m.T, 0, 1)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8)


def _stats(lab, w):
    w = w.astype(np.float64)[..., None]
    total = max(float(w.sum()), 1e-6)
    mu = (lab * w).sum((0, 1)) / total
    sd = np.sqrt(((lab - mu) ** 2 * w).sum((0, 1)) / total) + 1e-3
    return mu, sd


def match_subject(image, matte, direction="subject", strength=0.7, light=0.5, reference=None):
    """Photoshop-style Match Color between a picture's subject and its background (or another picture).

    matte: the subject (H, W) floats 0..1 (✂ Cut out's finder). direction "subject": the subject takes on the light and
    colour of the background right around it (a pasted or AI-made person that doesn't sit in the scene);
    "background": the background takes on the subject's. reference: a PIL picture whose overall colours to match
    instead. Lab mean / spread transfer (Reinhard), spreads limited to ×0.6–1.6 so nothing goes extreme; light (0..1)
    is how much of the brightness change is used (colour is used fully), strength how much of the change is kept.
    -> (PIL picture, {"moved": average change in Lab units})"""
    rgb = np.asarray(image.convert("RGB"))
    lab = _to_lab(rgb)
    m = np.clip(np.asarray(matte, np.float32), 0, 1)
    if m.shape != lab.shape[:2]:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR), np.float32) / 255
    if reference is not None:
        ref = reference.convert("RGB")
        ref.thumbnail((512, 512))
        rlab = _to_lab(np.asarray(ref))
        target = _stats(rlab, np.ones(rlab.shape[:2], np.float32))
    if direction == "background":
        source_w, apply_w = 1 - m, 1 - m
        if reference is None:
            target = _stats(lab, m)
    else:
        source_w, apply_w = m, m
        if reference is None:
            # the background close around the subject (its light is what the subject should share), not the far sky
            ring = _blur(m, max(4.0, 0.06 * max(m.shape))) * (1 - m)
            if ring.sum() < 1e-3 * m.size:
                ring = 1 - m
            target = _stats(lab, ring)
    if source_w.sum() < 1e-3 * m.size:
        raise ValueError("no subject found")
    mu_s, sd_s = _stats(lab, source_w)
    mu_t, sd_t = target
    scale = np.clip(sd_t / sd_s, 0.6, 1.6)
    new = (lab - mu_s) * scale + mu_t
    keep = np.array([float(light), 1.0, 1.0], np.float32)  # brightness gently, colour fully
    delta = (new - lab) * keep * float(strength) * apply_w[..., None]
    out = _from_lab(lab + delta)
    moved = float(np.abs(delta).sum(-1)[apply_w > 0.5].mean()) if (apply_w > 0.5).any() else 0.0
    return Image.fromarray(out), {"moved": round(moved, 2)}
