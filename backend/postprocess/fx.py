"""✨ Effects lab (Tools + the patchbay's FX module): three photo effects worked out on the picture's pixels (no model,
no download, a second or two), the kind of looks people buy as Photoshop actions:

- sparkler: paint a path round a person or a thing and it becomes a long-exposure sparkler trail: a white-hot core,
  a warm glow and sparks shooting off it (the scene can be darkened to night first);
- engrave: the picture as a banknote-style line engraving (lines that thicken in the shadows and bend with the tones,
  cross-hatched in the darkest parts), or as halftone dots;
- glass: part of the picture seen through a panel of fluted (reeded) glass: thin vertical ribs that smear and shift
  what's behind them, with faint colour fringes, in a shape you place.
"""
import numpy as np

SPARK_COLOURS = {"gold": ((1.0, 0.78, 0.38), (1.0, 0.55, 0.15)), "white": ((1.0, 0.97, 0.9), (0.85, 0.85, 1.0)),
                 "blue": ((0.75, 0.9, 1.0), (0.25, 0.55, 1.0)), "pink": ((1.0, 0.8, 0.92), (1.0, 0.3, 0.7)),
                 "green": ((0.85, 1.0, 0.8), (0.3, 1.0, 0.45))}
INKS = {"black": (0.07, 0.07, 0.08), "sepia": (0.28, 0.17, 0.08), "blue": (0.08, 0.13, 0.32), "red": (0.45, 0.07, 0.06)}
PAPER = (0.97, 0.96, 0.93)
SHAPES = ("diamond", "circle", "arch", "band", "all")


def _f(im):
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _out(x, alpha=None):
    from PIL import Image
    im = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    if alpha is not None:
        im.putalpha(alpha)
    return im


def _blur(a, sigma):
    import cv2
    if sigma <= 0:
        return a
    return cv2.GaussianBlur(np.ascontiguousarray(a, np.float32), (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def thin(mask):
    """Zhang-Suen thinning: a painted stroke (bool H×W) -> its one-pixel centre line."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = img
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]  # P2..P9
            c = p[1:-1, 1:-1]
            b = sum(n)
            a = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                m1, m2 = n[0] * n[2] * n[4], n[2] * n[4] * n[6]
            else:
                m1, m2 = n[0] * n[2] * n[6], n[0] * n[4] * n[6]
            kill = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & (m1 == 0) & (m2 == 0)
            if kill.any():
                img[1:-1, 1:-1][kill] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def sparkler(image, mask, colour="gold", glow=1.0, sparks=1.0, night=True, light=1.0, seed=11):
    """The painted path (a PIL mask, white = painted) -> a sparkler light trail along its centre line."""
    import cv2
    from PIL import Image
    x = _f(image)
    h, w = x.shape[:2]
    side = min(h, w)
    m = np.asarray(mask.convert("L").resize((w, h), Image.BILINEAR)) > 127
    if not m.any():
        raise ValueError("Paint the path for the sparkler first")
    # work on the centre line (thinned at a smaller size: quicker, and a thick brush still gives one line)
    k = min(1.0, 640 / max(h, w))
    small = cv2.resize(m.astype(np.uint8), (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_NEAREST) > 0
    line = thin(small)
    line = cv2.resize(line.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    line = np.clip(_blur(line, max(0.6, side / 1400)) * 3, 0, 1)
    albedo = x.copy()  # the surfaces' own colours, for the light the trail throws on them
    if night:  # long exposure at night: the scene sinks into a cool dark, the subject still just visible
        lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        x = (x * 0.28 + lum[..., None] * 0.04) * np.array([0.8, 0.88, 1.08], np.float32)
    hot, warm = (np.array(c, np.float32) for c in SPARK_COLOURS.get(colour, SPARK_COLOURS["gold"]))
    if light > 0:  # the trail lights what's near it: faces, clothes and the ground take its colour, brightest close by
        near = _blur(line, side / 14) + 0.6 * _blur(line, side / 6)
        near = np.clip(near / max(1e-4, float(np.percentile(near[line > 0.5], 60)) if (line > 0.5).any() else 1.0), 0, 1.6)
        x = x + albedo * warm * (near[..., None] ** 1.2) * 0.9 * float(light)
    light = np.zeros_like(x)
    light += line[..., None] * hot * 1.6                                                   # white-hot core
    for s, a in ((side / 260, 1.2), (side / 90, 0.8), (side / 30, 0.45)):                  # glow, near to far
        light += _blur(line, s)[..., None] * warm * a * 2.2 * float(glow)
    # sparks: short bright streaks shooting off the line, more where the line is
    ys, xs = np.nonzero(line > 0.5)
    rng = np.random.default_rng(seed)
    if len(xs) and sparks > 0:
        canvas = np.zeros((h, w), np.float32)
        n = int(min(6000, len(xs) * 0.35 * float(sparks)))
        pick = rng.integers(0, len(xs), n)
        for i in pick:
            ang = rng.uniform(0, 2 * np.pi)
            length = side * rng.uniform(0.006, 0.045) * (1 + rng.exponential(0.6))
            x0, y0 = xs[i], ys[i]
            x1, y1 = int(x0 + np.cos(ang) * length), int(y0 + np.sin(ang) * length)
            cv2.line(canvas, (int(x0), int(y0)), (x1, y1), float(rng.uniform(0.35, 1.0)), 1, cv2.LINE_AA)
            if rng.random() < 0.25:  # a tiny glowing dot at the end
                cv2.circle(canvas, (x1, y1), 1, 1.0, -1, cv2.LINE_AA)
        light += canvas[..., None] * hot * 1.2 + _blur(canvas, side / 500)[..., None] * warm * 1.5
    out = 1 - (1 - x) * (1 - np.clip(light, 0, 1))  # screen: light adds, never darkens
    return _out(out)


def engrave(image, style="lines", spacing=1.0, angle=45, ink="black", cross=True, contrast=1.0):
    """A banknote-style line engraving (style "lines") or halftone dots ("dots"), ink on paper."""
    from PIL import Image
    alpha = image.getchannel("A") if image.mode in ("RGBA", "LA") else None
    x = _f(image)
    h, w = x.shape[:2]
    side = min(h, w)
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)  # spread the tones (a dark photo isn't all ink)
    lum = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)
    lum = np.clip((lum - 0.5) * float(contrast) + 0.5, 0, 1) ** 0.75  # a little lighter: engravings are mostly paper
    dark = (1 - lum) ** 1.25
    period = max(3.0, side / 115 * float(spacing))  # line spacing in pixels
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if style == "dots":
        t = np.deg2rad(angle)
        u = (xx * np.cos(t) + yy * np.sin(t)) / period
        v = (-xx * np.sin(t) + yy * np.cos(t)) / period
        d = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2)  # distance to the nearest dot centre
        r = np.sqrt(_blur(dark, period / 3) / np.pi) * 1.05            # dot radius for that much ink
        ink_amt = np.clip((r - d) * period * 1.5 + 0.5, 0, 1)
    else:
        smooth = _blur(lum, period * 1.5)
        def screen(a, bend):
            t = np.deg2rad(a)
            u = (xx * np.cos(t) + yy * np.sin(t)) / period + smooth * bend  # the lines bend with the tones
            ph = np.abs(u - np.round(u)) * 2                                # 0 at a line's centre, 1 between
            width = np.clip(dark, 0, 1) ** 0.9                             # thicker in the shadows
            return np.clip((width - ph) * period * 0.9 + 0.5, 0, 1)
        ink_amt = screen(angle, 2.2)
        if cross:  # the darkest parts get a second set of lines across the first
            second = screen(angle + 90, -1.6) * np.clip((dark - 0.62) * 3, 0, 1)
            ink_amt = np.maximum(ink_amt, second)
    paper, inkc = np.array(PAPER, np.float32), np.array(INKS.get(ink, INKS["black"]), np.float32)
    out = paper + (inkc - paper) * ink_amt[..., None]
    return _out(out, alpha)


def shape_mask(h, w, shape="diamond", cx=0.35, cy=0.5, size=0.55):
    """The glass panel's shape: a soft-edged 0..1 mask. cx, cy: its centre, size: of the shorter side (0..1)."""
    import cv2
    m = np.zeros((h, w), np.float32)
    s = min(h, w) * float(size)
    c = (float(cx) * w, float(cy) * h)
    if shape == "all":
        m[:] = 1
    elif shape == "circle":
        cv2.circle(m, (int(c[0]), int(c[1])), int(s / 2), 1.0, -1, cv2.LINE_AA)
    elif shape == "band":  # a tall band from top to bottom
        m[:, max(0, int(c[0] - s / 4)):min(w, int(c[0] + s / 4))] = 1
    elif shape == "arch":
        x0, x1, y1 = int(c[0] - s / 3), int(c[0] + s / 3), int(c[1] + s / 2)
        cv2.rectangle(m, (x0, int(c[1] - s / 6)), (x1, y1), 1.0, -1)
        cv2.circle(m, (int(c[0]), int(c[1] - s / 6)), int(s / 3), 1.0, -1, cv2.LINE_AA)
    else:  # diamond: a rounded square turned 45°
        r = s / 2
        pts = np.array([(c[0], c[1] - r), (c[0] + r, c[1]), (c[0], c[1] + r), (c[0] - r, c[1])], np.int32)
        cv2.fillConvexPoly(m, pts, 1.0, cv2.LINE_AA)
        m = np.clip(_blur(m, s / 40) * 1.6 - 0.3, 0, 1)  # rounded corners
    return np.clip(_blur(m, max(0.8, min(h, w) / 900)), 0, 1)


def glass(image, shape="diamond", cx=0.35, cy=0.5, size=0.55, rib=1.0, strength=1.0, mask=None):
    """Part of the picture seen through fluted (reeded) glass: in a shape, or (mask: a PIL picture, white = glass) in
    an area you painted or selected (a box, the subject, its inverse…)."""
    import cv2
    alpha = image.getchannel("A") if image.mode in ("RGBA", "LA") else None
    x = _f(image)
    h, w = x.shape[:2]
    side = min(h, w)
    rw = max(4.0, side / 45 * float(rib))                      # rib width in pixels
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    u = (xx % rw) / rw                                          # 0..1 across each rib
    shift = (0.5 - u) * rw * 2.2 * float(strength)              # each rib magnifies and flips its strip
    def warp(dx):
        return cv2.remap(x, xx + shift + dx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fr = rw * 0.08 * float(strength)                            # faint colour fringes
    g = np.stack([warp(fr)[..., 0], warp(0)[..., 1], warp(-fr)[..., 2]], -1)
    k = max(3, int(side / 40) | 1)
    g = cv2.blur(g, (1, k))                                     # glass smears up and down a little
    g = g * (0.88 + 0.2 * np.sin(np.pi * u))[..., None] + 0.04  # each rib's rounded face catches the light
    edge = np.exp(-((u - 0.06) / 0.04) ** 2) * 0.12             # a thin bright line along each rib
    g = np.clip(g + edge[..., None], 0, 1)
    if mask is not None:
        from PIL import Image
        m = np.asarray(mask.convert("L").resize((w, h), Image.BILINEAR), np.float32) / 255.0
        if m.max() < 0.5:
            raise ValueError("Paint (or select) where the glass goes first")
        m = np.clip(_blur(m, max(0.8, side / 900)), 0, 1)
    else:
        m = shape_mask(h, w, shape, cx, cy, size)
    rim = np.clip(_blur(m, 1.0) - _blur(m, 4.0), 0, 1) * 1.4    # the panel's polished edge
    out = x + (g - x) * m[..., None]
    out = np.clip(out + rim[..., None] * 0.5, 0, 1)
    return _out(out, alpha)
