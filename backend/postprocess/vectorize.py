"""✒️ Vectorize: turn a picture into an SVG made of flat colour shapes (like Vector Magic or Illustrator's Image
Trace), for logos, icons, stickers, cut files and flat illustrations.

How: the colours are reduced to a small palette (k-means), each colour's areas are cleaned of specks, their
outlines are traced (with the holes in them), simplified, and drawn as smooth curves. The biggest areas are
painted first, so smaller shapes sit on top. Pure OpenCV + numpy: no download.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DETAIL = {  # detail level -> (simplify tolerance as a fraction of the size, smallest shape in pixels² at 1000 px)
    "low": (0.0030, 160),
    "medium": (0.0016, 60),
    "high": (0.0008, 18),
}


def _palette(rgb, colors, seed=1):
    import cv2
    data = rgb.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(seed)
    _c, labels, centers = cv2.kmeans(data, colors, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(rgb.shape[:2]), np.clip(centers, 0, 255).astype(np.uint8)


def _smooth_path(pts, smooth, on_edge=lambda p: False):
    """Closed path through pts. smooth=True: quadratic curves through the midpoints (rounded, like a pen tool),
    except at points on the picture's edge, which stay sharp (so shapes still fill the corners);
    False: straight lines (crisp, pixel-art and geometric logos)."""
    n = len(pts)
    if n < 3:
        return ""
    f = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
    if not smooth:
        return "M" + " L".join(f"{f(x)} {f(y)}" for x, y in pts) + " Z"
    mids = [((pts[i][0] + pts[(i + 1) % n][0]) / 2, (pts[i][1] + pts[(i + 1) % n][1]) / 2) for i in range(n)]
    d = [f"M{f(mids[-1][0])} {f(mids[-1][1])}"]
    for i in range(n):
        if on_edge(pts[i]):
            d.append(f"L{f(pts[i][0])} {f(pts[i][1])} L{f(mids[i][0])} {f(mids[i][1])}")
        else:
            d.append(f"Q{f(pts[i][0])} {f(pts[i][1])} {f(mids[i][0])} {f(mids[i][1])}")
    return " ".join(d) + " Z"


def vectorize(image, colors=8, detail="medium", smooth=True, background=True, max_side=1200):
    """image: PIL. Returns (svg_text, stats). colors: 2-32 (the palette size). detail: low / medium / high."""
    import cv2
    colors = int(max(2, min(32, colors)))
    eps_frac, min_area_1000 = DETAIL.get(detail, DETAIL["medium"])
    src = image.convert("RGBA")
    W, H = src.size
    scale = min(1.0, max_side / max(W, H))
    work = src.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.LANCZOS) if scale < 1 else src
    rgba = np.asarray(work)
    alpha = rgba[..., 3] > 127
    rgb = cv2.bilateralFilter(np.ascontiguousarray(rgba[..., :3]), 7, 40, 7)  # flatten noise, keep edges
    labels, centers = _palette(rgb, colors)
    labels = np.where(alpha, labels, -1)
    h, w = labels.shape
    side = max(w, h)
    eps = max(0.35, eps_frac * side)
    min_area = min_area_1000 * (side / 1000.0) ** 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # paint order: the background colour (the one most common along the picture's border) first, then the
    # biggest areas, so smaller shapes sit on top
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border = border[border >= 0]
    bg = int(np.bincount(border).argmax()) if border.size else None
    order = sorted(range(len(centers)), key=lambda k: (k != bg, -int((labels == k).sum())))
    inv = 1.0 / scale
    layers, shapes = [], 0
    for rank, k in enumerate(order):
        m = (labels == k).astype(np.uint8) * 255
        if not m.any():
            continue
        if rank == 0 and k == bg and background and alpha.all():
            r, g, b = (int(c) for c in centers[k])
            layers.append(f'<rect width="{W}" height="{H}" fill="#{r:02x}{g:02x}{b:02x}"/>')
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is None:
            continue
        hier = hier[0]
        parts = []
        for i, c in enumerate(contours):
            if hier[i][3] != -1:
                continue  # holes are drawn with their outer shape
            if cv2.contourArea(c) < min_area:
                continue
            group = [c]
            j = hier[i][2]
            while j != -1:
                if cv2.contourArea(contours[j]) >= min_area:
                    group.append(contours[j])
                j = hier[j][0]
            for cc in group:
                ap = cv2.approxPolyDP(cc, eps, True).reshape(-1, 2)
                pts = [(float(x) * inv, float(y) * inv) for x, y in ap]
                tol = 1.5 * inv
                p = _smooth_path(pts, smooth, lambda q: q[0] <= tol or q[1] <= tol or q[0] >= (w - 1) * inv - tol
                                 or q[1] >= (h - 1) * inv - tol)
                if p:
                    parts.append(p)
            shapes += 1
        if parts:
            r, g, b = (int(c) for c in centers[k])
            layers.append(f'<path fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd" d="{" ".join(parts)}"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
           f'<!-- Made with Protoart Studio · Vectorize ({colors} colours, {detail} detail) -->'
           + "".join(layers) + "</svg>")
    return svg, {"colors": len(centers), "shapes": shapes, "bytes": len(svg)}
