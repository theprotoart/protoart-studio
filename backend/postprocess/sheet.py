"""🧩 Split a character sheet: find each figure on a sheet (a grid of poses, stickers, expressions…) and cut it out on
its own. Pure numpy / OpenCV here (the background finder runs in engine._cut_sheet), so it's testable without a model."""
import numpy as np


def _cv2():
    import cv2
    return cv2


def reading_order(boxes):
    """sort (x0, y0, x1, y1) boxes in rows, top to bottom, then left to right"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    rows, row, row_y = [], [], None
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        if row and abs(cy - row_y) > med_h * 0.5:
            rows.append(row)
            row = []
        row.append(b)
        row_y = float(np.mean([(r[1] + r[3]) / 2 for r in row]))
    rows.append(row)
    return [b for r in rows for b in sorted(r, key=lambda b: b[0])]


def grid_boxes(width, height, rows, cols):
    """the cells of an even rows × cols grid"""
    rows, cols = max(1, int(rows)), max(1, int(cols))
    xs = np.linspace(0, width, cols + 1).round().astype(int)
    ys = np.linspace(0, height, rows + 1).round().astype(int)
    return [(int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1])) for r in range(rows) for c in range(cols)]


def split_big(boxes):
    """figures that touch come out as one big box: split a box that's about 2, 3… figures wide or tall evenly"""
    if len(boxes) < 3:
        return boxes
    mw = float(np.median([b[2] - b[0] for b in boxes]))
    mh = float(np.median([b[3] - b[1] for b in boxes]))
    out = []
    for b in boxes:
        nx = max(1, int(round((b[2] - b[0]) / mw))) if (b[2] - b[0]) > mw * 1.6 else 1
        ny = max(1, int(round((b[3] - b[1]) / mh))) if (b[3] - b[1]) > mh * 1.6 else 1
        if nx == 1 and ny == 1:
            out.append(b)
            continue
        for x0, y0, x1, y1 in grid_boxes(b[2] - b[0], b[3] - b[1], ny, nx):
            out.append((b[0] + x0, b[1] + y0, b[0] + x1, b[1] + y1))
    return out


def profile_cuts(mask, axis, min_seg=0.06, depth=0.45):
    """where the figures' rows (axis 1) or columns (axis 0) part: the low points of how much figure there is across
    the sheet. Works when neighbours touch here and there (a sleeve over the next head), as long as most of the line
    between them is background."""
    prof = (mask > 0.5).mean(axis=axis).astype(np.float64)
    n = len(prof)
    win = max(3, int(n * 0.015))
    prof = np.convolve(prof, np.ones(win) / win, mode="same")
    seg = max(2, int(n * min_seg))
    minima = [i for i in range(seg, n - seg) if prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]]
    cuts = []
    for i in sorted(minima, key=lambda i: prof[i]):
        if any(abs(i - c) < seg for c in cuts):
            continue
        a = max([c for c in cuts if c < i] or [0])
        b = min([c for c in cuts if c > i] or [n])
        if i - a < seg or b - i < seg:
            continue
        if prof[i] < depth * min(prof[a:i].max(), prof[i:b].max()):
            cuts.append(i)
    return sorted(cuts)


def _even(edges):
    """a sheet's columns (or rows) are about the same size: a strip under half the usual one is joined to its
    smaller neighbour (a stray low point inside a figure, or a figure cut off at the sheet's edge)"""
    edges = list(edges)
    while len(edges) > 3:
        sizes = np.diff(edges)
        med = float(np.median(sizes))
        i = int(np.argmin(sizes))
        if sizes[i] >= med * 0.5:
            break
        if i == 0:
            del edges[1]
        elif i == len(sizes) - 1:
            del edges[-2]
        else:
            del edges[i if sizes[i - 1] <= sizes[i + 1] else i + 1]
    return edges


def profile_boxes(mask):
    """the cells between the low points across and down the sheet (empty ones come out empty when cut)"""
    h, w = mask.shape
    xs = _even([0] + profile_cuts(mask, 0) + [w])
    ys = _even([0] + profile_cuts(mask, 1) + [h])
    boxes = []
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            if (mask[y0:y1, x0:x1] > 0.5).mean() > 0.03:
                boxes.append((x0, y0, x1, y1))
    return boxes


def _period(prof, lo_frac=0.08, hi_frac=0.7):
    """the repeat distance of a sheet's rows (or columns), from how the figure coverage repeats -> (lag, strength)"""
    x = prof - prof.mean()
    n = len(x)
    if n < 16 or not x.any():
        return 0, 0.0
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / max(1e-9, ac[0])
    lo, hi = max(2, int(n * lo_frac)), max(3, int(n * hi_frac))
    best, score = 0, 0.0
    for lag in range(lo, min(hi, n - 1)):
        if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > score:
            best, score = lag, float(ac[lag])
    return best, score


def grid_cuts(mask, axis, strong=0.25):
    """the sheet's even rows (axis 1) or columns (axis 0): how many there are from the repeat distance, each cut moved
    to the emptiest line near where it's expected. Works when the figures overlap their neighbours (no clear gap)."""
    prof = (mask > 0.5).mean(axis=axis).astype(np.float64)
    n = len(prof)
    win = max(3, int(n * 0.01))
    prof = np.convolve(prof, np.ones(win) / win, mode="same")
    lag, score = _period(prof)
    if not lag or score < strong:
        return []
    k = int(round(n / lag))
    if k < 2:
        return []
    cuts = []
    for i in range(1, k):
        c = int(round(i * n / k))
        r = max(1, int(n / k * 0.2))
        a, b = max(1, c - r), min(n - 1, c + r)
        cuts.append(a + int(np.argmin(prof[a:b])))
    return cuts


def grid_boxes_auto(mask):
    """the cells of the sheet's own grid (found from its repeat), empty ones left out"""
    h, w = mask.shape
    xs = [0] + grid_cuts(mask, 0) + [w]
    ys = [0] + grid_cuts(mask, 1) + [h]
    return [(x0, y0, x1, y1) for y0, y1 in zip(ys, ys[1:]) for x0, x1 in zip(xs, xs[1:])
            if (mask[y0:y1, x0:x1] > 0.5).mean() > 0.03]


def find_figures(mask, min_frac=0.15):
    """mask: (h, w) 0..1, 1 = a figure. Returns one box per figure, in reading order. Bits smaller than min_frac of the
    biggest figure (specks, stray marks) are dropped; bits close together (a detached hand, a mug) are joined."""
    cv2 = _cv2()
    h, w = mask.shape
    solid = (mask > 0.5).astype(np.uint8)
    k = max(3, int(round(min(h, w) * 0.012)) | 1)
    joined = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= areas.max() * min_frac * 0.5]
    boxes = [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 0] + stats[i, 2]), int(stats[i, 1] + stats[i, 3])) for i in keep]
    # drop a box that's small next to the typical figure (after the big-figure check above, by size of box)
    if len(boxes) >= 3:
        med = float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
        boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= med * min_frac]
    boxes = split_big(boxes)
    # figures that touch merge into one piece: the sheet's own rows and columns part them (its gaps, or else how
    # its figures repeat across and down); a grid that finds more figures than the pieces wins
    for cells in (profile_boxes(mask), grid_boxes_auto(mask)):
        if len(cells) > len(boxes) * 1.2:
            boxes = cells
    return reading_order(boxes)


def grow_box(box, width, height, frac=0.08):
    """the box with a margin around it (inside the picture), so arms and hair that stick out aren't cut off"""
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * frac), int((y1 - y0) * frac)
    return (max(0, x0 - mx), max(0, y0 - my), min(width, x1 + mx), min(height, y1 + my))


def keep_main(alpha, inner, edges=(True, True, True, True)):
    """alpha: the crop's (h, w) 0..1 cut-out; inner: the figure's own cell inside the crop (x0, y0, x1, y1).
    Keeps the parts that are mostly inside the cell (the figure, with a detached hand or prop) and drops the bits of
    the neighbours that reach into the margin. edges: which sides of the crop are inside the sheet (left, top, right,
    bottom); a piece touching one of those is part of a neighbour (unless it's the figure itself)."""
    cv2 = _cv2()
    solid = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if n <= 1:
        return alpha
    x0, y0, x1, y1 = inner
    inside = np.zeros_like(solid)
    inside[y0:y1, x0:x1] = 1
    keep = np.zeros(n, bool)
    best, best_in = 0, -1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        within = int((inside[labels == i]).sum())
        if within > best_in:
            best, best_in = i, within
        # a piece cut off at the crop's edge goes on outside it: a neighbour's hand or sleeve reaching in
        l, t, cw, ch = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        at_edge = (l == 0 and edges[0]) or (t == 0 and edges[1]) or (l + cw >= solid.shape[1] and edges[2]) or (t + ch >= solid.shape[0] and edges[3])
        # the figure's own detached bits (a mug, a hand) are inside its cell; a piece that crosses the cell's edge is
        # a neighbour reaching in
        if within >= area * 0.9 and area >= solid.size * 0.002 and not at_edge:
            keep[i] = True
    keep[best] = True
    gone = (~keep[labels]) & (labels > 0)
    gone = cv2.dilate(gone.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # don't eat into what we keep with the dilation
    gone &= ~keep[labels]
    out = alpha * (~gone)
    # a neighbour that touches the figure is joined to it: what's outside the cell and runs on to the crop's edge is
    # the neighbour's (the figure's own hand or hair sticking out stops before the edge)
    outside = np.ones_like(solid)
    outside[y0:y1, x0:x1] = 0
    bits = ((out > 0.5) & outside.astype(bool)).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(bits, 8)
    H, W = solid.shape
    for i in range(1, n2):
        l, t, cw, ch = st2[i, 0], st2[i, 1], st2[i, 2], st2[i, 3]
        if (l == 0 and edges[0]) or (t == 0 and edges[1]) or (l + cw >= W and edges[2]) or (t + ch >= H and edges[3]):
            out[lab2 == i] = 0
    # faint see-through leftovers (below the 0.5 the pieces are counted at) away from what's kept: slivers of the
    # neighbours' soft edges along the crop
    kept = cv2.dilate((out > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return out * kept


def tight_box(alpha, pad):
    ys, xs = np.nonzero(alpha > 0.02)
    if not len(xs):
        return None
    h, w = alpha.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


def on_square(image, pad_frac=0.04):
    """the picture centred on a square see-through (or its own) canvas, for same-size stickers"""
    from PIL import Image
    side = int(max(image.size) * (1 + pad_frac * 2))
    canvas = Image.new(image.mode, (side, side), (0, 0, 0, 0) if image.mode == "RGBA" else (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas
