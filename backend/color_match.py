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
