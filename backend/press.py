"""Press: the smallest JPEG / PNG that still measures as looking the same (ported from ProtoArt Press).

Every candidate is compared with the original using a windowed SSIM on brightness (8×8 windows, 4 px apart),
looking at the average, the worst 64×64 region and the largest brightness shift in any region (SSIM alone forgives
flat shifts; this catches banding). Anything that wouldn't look the same isn't used.

  * JPEG: binary search for the lowest quality whose damage stays within this picture's own budget, measured
    against its near-lossless baseline (quality 96): Gentle ≈ q90, Balanced ≈ q85, Strong ≈ q80 on typical photos.
  * PNG: lossless repacking always (palette / grey when exact, maximum zlib); with Balanced or Strong also a
    256-colour version when it measures as invisible. What's written is decoded again and checked pixel for pixel.
Pure Python + numpy + Pillow; runs on this Mac only.
"""
import io

import numpy as np

STRENGTHS = ("gentle", "balanced", "strong")
JPEG_ALLOWANCE = {"gentle": 10, "balanced": 15, "strong": 21}
JPEG_BASELINE_Q, JPEG_LOWEST_Q, JPEG_STEPS = 96, 30, 7
PNG_LIMITS = {
    "gentle": None,  # lossless only
    "balanced": {"min_mean": 0.995, "min_worst": 0.985, "max_shift": 1.0},
    "strong": {"min_mean": 0.990, "min_worst": 0.975, "max_shift": 1.5},
}


# ------------------------------------------------------------------ measuring
def luma(img, background=255):
    """Brightness plane (float64) of a PIL image, transparent parts composited over grey `background`."""
    a = np.asarray(img.convert("RGBA"), dtype=np.float64)
    y = (77 * a[..., 0] + 150 * a[..., 1] + 29 * a[..., 2] + 128) / 256.0
    alpha = a[..., 3] / 255.0
    return np.floor(y * alpha + background * (1 - alpha))


def _window_sums(x, win=8, step=4):
    """Sums over win×win windows placed every `step` px (via an integral image)."""
    ii = np.zeros((x.shape[0] + 1, x.shape[1] + 1))
    ii[1:, 1:] = x.cumsum(0).cumsum(1)
    ys = np.arange(0, x.shape[0] - win + 1, step)
    xs = np.arange(0, x.shape[1] - win + 1, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return ii[Y + win, X + win] - ii[Y, X + win] - ii[Y + win, X] + ii[Y, X]


def compare(a, b):
    """-> {"mean", "worst_region", "worst_shift"} for two brightness planes of the same size."""
    if a.shape[0] < 8 or a.shape[1] < 8:
        return {"mean": 1.0, "worst_region": 1.0, "worst_shift": 0.0}
    n = 64.0
    sa, sb = _window_sums(a), _window_sums(b)
    saa, sbb, sab = _window_sums(a * a), _window_sums(b * b), _window_sums(a * b)
    ma, mb = sa / n, sb / n
    va, vb, cv = saa / n - ma * ma, sbb / n - mb * mb, sab / n - ma * mb
    c1, c2 = 6.5025, 58.5225
    ssim = ((2 * ma * mb + c1) * (2 * cv + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    shift = np.abs(ma - mb)
    # regions of 16×16 windows (≈ 64×64 px); regions at least half full count, like Press
    ny, nx = ssim.shape
    worst, worst_shift, worst_any, shift_any, saw_full = 1.0, 0.0, 1.0, 0.0, False
    for ty in range(0, ny, 16):
        for tx in range(0, nx, 16):
            blk = ssim[ty:ty + 16, tx:tx + 16]
            sh = shift[ty:ty + 16, tx:tx + 16].mean()
            avg = blk.mean()
            worst_any, shift_any = min(worst_any, avg), max(shift_any, sh)
            if blk.size >= 128:
                saw_full = True
                worst, worst_shift = min(worst, avg), max(worst_shift, sh)
    if not saw_full:
        worst, worst_shift = worst_any, shift_any
    return {"mean": float(ssim.mean()), "worst_region": float(worst), "worst_shift": float(worst_shift)}


# ------------------------------------------------------------------ JPEG
def _encode_jpeg(img, q, **kw):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(q), optimize=True, progressive=True, **kw)
    return buf.getvalue()


def smallest_jpeg(img, strength="balanced", **save_kw):
    """-> (bytes, quality). img: RGB PIL image. save_kw: extra JPEG options (dpi, icc_profile, comment…)."""
    from PIL import Image
    img = img.convert("RGB")
    ref = luma(img)

    def score(data):
        with Image.open(io.BytesIO(data)) as d:
            return compare(ref, luma(d))

    base = _encode_jpeg(img, JPEG_BASELINE_Q, **save_kw)
    s0 = score(base)
    k = JPEG_ALLOWANCE.get(strength, JPEG_ALLOWANCE["balanced"])
    max_mean = max((1 - s0["mean"]) * k, 0.002)
    max_region = max((1 - s0["worst_region"]) * k, 0.004)
    best, best_q = base, JPEG_BASELINE_Q
    lo, hi = float(JPEG_LOWEST_Q), float(JPEG_BASELINE_Q)
    for _ in range(JPEG_STEPS):
        mid = (lo + hi) / 2
        data = _encode_jpeg(img, round(mid), **save_kw)
        s = score(data)
        if 1 - s["mean"] <= max_mean and 1 - s["worst_region"] <= max_region:
            if len(data) < len(best):
                best, best_q = data, round(mid)
            hi = mid
        else:
            lo = mid
    return best, best_q


# ------------------------------------------------------------------ PNG
def _encode_png(img, **kw):
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True, compress_level=9, **kw)
    return buf.getvalue()


def _lossless_variant(img):
    """Exact smaller representation: grey, or a palette (with per-colour transparency) when there are at most
    256 different colours. The pixels stay exactly the same."""
    from PIL import Image
    rgba = np.ascontiguousarray(np.asarray(img.convert("RGBA")))
    h, w = rgba.shape[:2]
    opaque = bool((rgba[..., 3] == 255).all())
    rgb = rgba[..., :3]
    if opaque and (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all():
        return Image.fromarray(np.ascontiguousarray(rgb[..., 0]), "L")
    flat = rgba.reshape(-1, 4).view(np.uint32).ravel()
    colors = np.unique(flat)
    if colors.size <= 256:
        idx = np.searchsorted(colors, flat).astype(np.uint8).reshape(h, w)
        pal = colors.view(np.uint8).reshape(-1, 4)
        p = Image.fromarray(idx, "P")
        p.putpalette(pal[:, :3].ravel().tolist())
        if not opaque:
            p.info["transparency"] = bytes(pal[:, 3].tolist())
        return p
    return img.convert("RGB") if opaque else img.convert("RGBA")


def _looks_same(a, b, opaque, limits):
    for bg in ([255] if opaque else [255, 0]):
        s = compare(luma(a, bg), luma(b, bg))
        if s["mean"] < limits["min_mean"] or s["worst_region"] < limits["min_worst"] or s["worst_shift"] > limits["max_shift"]:
            return False
    return True


def smallest_png(img, strength="balanced", **save_kw):
    """-> (bytes, detail). Lossless always; a 256-colour version only when it measures as invisible."""
    from PIL import Image
    rgba = img.convert("RGBA")
    a = np.asarray(rgba).copy()
    a[a[..., 3] == 0, :3] = 0  # invisible pixels: same colour, so they compress better
    rgba = Image.fromarray(a, "RGBA")
    opaque = bool((a[..., 3] == 255).all())
    exact = _lossless_variant(rgba)
    kw = dict(save_kw)
    if exact.mode == "P" and "transparency" in exact.info:
        kw["transparency"] = exact.info["transparency"]
    if exact.mode == "L":
        kw.pop("icc_profile", None)  # an RGB colour profile doesn't belong on a grey picture
    cands = [(_encode_png(exact, **kw), "lossless", exact)]
    limits = PNG_LIMITS.get(strength)
    if limits and exact.mode not in ("P", "L"):
        src = rgba if not opaque else rgba.convert("RGB")
        reduced = src.quantize(colors=256, method=Image.Quantize.FASTOCTREE if not opaque else Image.Quantize.MEDIANCUT,
                               dither=Image.Dither.NONE)
        reduced_rgba = reduced.convert("RGBA")
        if _looks_same(rgba, reduced_rgba, opaque, limits):
            cands.append((_encode_png(reduced, **save_kw), "256 colours", reduced))
    cands.sort(key=lambda c: len(c[0]))
    for data, detail, used in cands:
        with Image.open(io.BytesIO(data)) as chk:  # decode what we wrote: exactly the pixels we meant?
            if chk.size == rgba.size and np.array_equal(np.asarray(chk.convert("RGBA")), np.asarray(used.convert("RGBA"))):
                return data, detail
    return cands[-1][0], cands[-1][1]
