"""🎞 Film looks (Tools): the colour and grain of five classic kinds of film stock, worked out on the picture's pixels
(no model, no download, about a second). Named by what they look like, not by brand.

Each look is: a tone curve (toe lift = faded blacks, contrast, a soft shoulder for the highlights), a colour balance
for the shadows and the highlights, saturation (with extra care for skin tones or blue skies), optional black and
white with its own colour filter, a vignette, halation (the red glow round bright lights on film with no anti-halation
layer) and grain that's strongest in the mid-tones, as real grain is.
"""
import numpy as np

# warm_portrait: a soft professional colour negative: low contrast, warm, kind to skin
# disposable: a 2000s consumer colour negative in a throwaway camera: punchy, green-cyan shadows,
#   a dark corner vignette, a little soft, more grain
# slide: a clean colour slide film: crisp, deep blacks, saturated, neutral to cool, fine grain
# gritty_bw: a classic fast black-and-white film, pushed: heavy grain, deep blacks, bright highlights
# night_tungsten: a tungsten-balanced motion-picture film with its anti-halation layer removed:
#   cool, teal shadows, red-orange halos round lights and bright edges, visible grain
STOCKS = {
    "warm_portrait": dict(label="Soft portrait", contrast=0.88, lift=0.035, shoulder=0.88, sat=0.94, skin=1.12,
                          greens=0.85, shadows=(0.012, 0.004, -0.010), highlights=(0.020, 0.010, -0.018),
                          grain=0.22, grain_size=1.1, halation=0.0, vignette=0.08, soften=0.0),
    "disposable": dict(label="Disposable camera", contrast=1.12, lift=0.035, shoulder=0.92, sat=1.12, skin=1.0,
                       greens=1.08, shadows=(-0.016, 0.018, 0.012), highlights=(0.018, 0.004, -0.008),
                       grain=0.42, grain_size=1.5, halation=0.12, vignette=0.32, soften=0.7),
    "slide": dict(label="Clean slide", contrast=1.18, lift=0.0, shoulder=0.96, sat=1.32, skin=0.96, greens=1.06,
                  shadows=(-0.006, -0.002, 0.012), highlights=(0.0, 0.002, 0.006),
                  grain=0.12, grain_size=0.9, halation=0.0, vignette=0.1, soften=0.0),
    "gritty_bw": dict(label="Gritty black & white", mono=(0.34, 0.50, 0.16), contrast=1.42, lift=0.0, shoulder=0.9,
                      sat=0.0, skin=1.0, greens=1.0, shadows=(0, 0, 0), highlights=(0, 0, 0),
                      grain=0.85, grain_size=1.7, halation=0.0, vignette=0.18, soften=0.0),
    "night_tungsten": dict(label="Night tungsten", contrast=1.06, lift=0.03, shoulder=0.86, sat=1.04, skin=1.0,
                           greens=0.92, shadows=(-0.040, 0.012, 0.060), highlights=(0.020, 0.006, -0.012),
                           grain=0.38, grain_size=1.35, halation=0.85, vignette=0.12, soften=0.0),
}
ORDER = ["warm_portrait", "disposable", "slide", "gritty_bw", "night_tungsten"]


def _curve(x, contrast, lift, shoulder):
    """Film's S-curve: contrast round the middle grey, a lifted toe (blacks never quite black) and a soft shoulder."""
    x = np.clip(x, 0.0, 1.0)
    mid = 0.45
    y = mid + (x - mid) * contrast
    # smooth roll-off instead of clipping: the highlights compress gently as they approach white
    over = np.maximum(y - shoulder, 0.0)
    head = max(1e-3, 1.0 - shoulder)
    y = np.where(y > shoulder, shoulder + head * (1.0 - np.exp(-over / head)), y)
    y = np.maximum(y, 0.0) * (1.0 - lift) + lift
    return np.clip(y, 0.0, 1.0)


def _hue_masks(rgb):
    """(skin tones, greens) as soft 0..1 masks from the hue."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    chroma = mx - mn + 1e-6
    hue = np.where(mx == r, ((g - b) / chroma) % 6, np.where(mx == g, (b - r) / chroma + 2, (r - g) / chroma + 4)) * 60
    sat = chroma / (mx + 1e-6)
    skin = np.exp(-((hue - 25) / 18) ** 2) * np.clip(sat * 3, 0, 1)
    greens = np.exp(-((hue - 110) / 35) ** 2) * np.clip(sat * 3, 0, 1)
    return skin, greens


def _blur(a, radius):
    """Gaussian blur of a float picture (H, W, 3); radius as Pillow's (about one standard deviation)."""
    import cv2
    if radius <= 0:
        return a
    return cv2.GaussianBlur(np.ascontiguousarray(a, dtype=np.float32), (0, 0), sigmaX=float(radius), borderType=cv2.BORDER_REFLECT)


def _grain(shape, size, seed):
    """Film grain: noise made smaller and blown up (bigger clumps for faster film), so it looks like grain, not pixels."""
    from PIL import Image
    h, w = shape
    rng = np.random.default_rng(seed)
    k = max(1.0, float(size))
    gh, gw = max(2, int(h / k)), max(2, int(w / k))
    n = rng.standard_normal((gh, gw)).astype(np.float32)
    im = Image.fromarray(n, mode="F").resize((w, h), Image.BICUBIC)
    n = np.asarray(im, dtype=np.float32)
    return n / (n.std() + 1e-6)


def apply(image, stock="warm_portrait", strength=1.0, grain=1.0, halation=1.0, seed=7):
    """PIL picture -> the same picture in that film's look. strength 0..1 (how much of the look), grain and halation
    0..2 (times the look's own amount)."""
    from PIL import Image
    p = STOCKS.get(stock)
    if p is None:
        raise ValueError(f"Unknown film look: {stock}")
    has_alpha = image.mode in ("RGBA", "LA")
    alpha = image.getchannel("A") if has_alpha else None
    src = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = src.shape[:2]
    side = min(h, w)
    x = src.copy()

    if p.get("soften"):  # a cheap plastic lens: a touch soft
        x = _blur(x, p["soften"] * side / 1000)

    # halation: bright parts glow red-orange into their surroundings (the light bounces off the film's back)
    amt = p["halation"] * float(halation)
    if amt > 0:
        lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        bright = np.clip((lum - 0.72) / 0.28, 0, 1)[..., None] ** 1.5
        glow = _blur(bright * np.array([1.0, 0.28, 0.08], np.float32), side / 55)
        glow += 0.6 * _blur(bright * np.array([1.0, 0.18, 0.05], np.float32), side / 160)
        x = 1 - (1 - x) * (1 - np.clip(glow * amt * 2.2, 0, 1))  # screen

    if p.get("mono"):
        y = x @ np.array(p["mono"], np.float32)
        x = np.repeat(y[..., None], 3, -1)
    else:
        skin, greens = _hue_masks(x)
        lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        s = p["sat"] * (1 + (p["skin"] - 1) * skin[..., None]) * (1 + (p["greens"] - 1) * greens[..., None])
        x = lum + (x - lum) * s

    x = _curve(x, p["contrast"], p["lift"], p["shoulder"])

    if not p.get("mono"):  # colour balance: one tint in the shadows, another in the highlights
        lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        x = x + np.array(p["shadows"], np.float32) * (1 - lum) ** 2 + np.array(p["highlights"], np.float32) * lum ** 2

    if p.get("vignette"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
        x = x * (1 - p["vignette"] * np.clip(d, 0, 1) ** 1.4)[..., None]

    g = p["grain"] * float(grain)
    if g > 0:
        lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        where = (0.25 + 4 * lum * (1 - lum))[..., None]  # most in the mid-tones, as on film
        n = _grain((h, w), p["grain_size"] * side / 1000, seed)[..., None]
        if not p.get("mono"):  # colour film: each layer has its own grain, mostly shared
            n = 0.8 * n + 0.2 * np.stack([_grain((h, w), p["grain_size"] * side / 1000, seed + i + 1) for i in range(3)], -1)
        x = x + n * where * g * 0.045

    x = np.clip(x, 0, 1)
    k = float(np.clip(strength, 0, 1))
    x = src + (x - src) * k
    out = Image.fromarray((x * 255 + 0.5).astype(np.uint8), "RGB")
    if has_alpha:
        out.putalpha(alpha)
    return out
