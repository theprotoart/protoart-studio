"""✂ Cut out (remove the background): finds the people, animals or things in front and makes everything else
see-through, for PNGs, stickers and Puppet warp.

The model is BiRefNet lite (MIT licence, ZhengPeng7/BiRefNet; ONNX export by onnx-community), ~224 MB, downloaded
once from Hugging Face at a pinned revision (huggingface_hub checks the file), then offline. It runs on the CPU
with ONNX Runtime: a few seconds, and no image model has to be loaded.

The model gives a "matte": how sure it is, for every pixel, that the pixel belongs to the subject (0..1). The
refining controls work on that matte, the same way in the window's live preview and in the saved file:
  threshold  where "subject" starts (lower keeps more: wisps of hair, a tail; higher drops faint bits)
  softness   how gradual the edge is (0 = a hard cut, like a sticker)
  grow       grow (+) or shrink (-) the edge by that many pixels (shrink takes away a halo of old background)
and, when saving, two clean-ups: small separate bits are dropped, and the edge's colours are taken from inside the
subject, so no fringe of the old background is left around it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

REPO = "onnx-community/BiRefNet_lite-ONNX"
REVISION = "de15b22ba131738a16dff04aab8bdf8dc32e3ac1"   # pinned: the file can't change under us
FILE = "onnx/model.onnx"
MB = 224
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_session = None


def session(download=True):
    global _session
    if _session is None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        path = hf_hub_download(REPO, FILE, revision=REVISION, local_files_only=not download)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def downloaded():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, FILE, revision=REVISION, local_files_only=True)
        return True
    except Exception:
        return False


def matte(image, sess=None):
    """image: PIL. Returns a float32 array (h, w) in 0..1: 1 = the subject."""
    sess = sess or session()
    rgb = image.convert("RGB")
    w, h = rgb.size
    x = np.asarray(rgb.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    inp = sess.get_inputs()[0]
    outs = sess.run(None, {inp.name: x})
    out = outs[-1] if outs[-1].ndim == 4 else outs[0]   # the finest prediction
    m = np.asarray(out, np.float32).reshape(out.shape[-2], out.shape[-1])
    if m.min() < -0.01 or m.max() > 1.01:   # logits -> 0..1
        m = 1.0 / (1.0 + np.exp(-np.clip(m, -30, 30)))
    big = Image.fromarray(np.clip(m * 255, 0, 255).astype(np.uint8), "L").resize((w, h), Image.BILINEAR)
    return np.asarray(big, np.float32) / 255.0


def _slide(a, r, fn, axis):
    """max (grow) or min (shrink) over a window of 2r+1 pixels along one axis, like the window's preview"""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    f = maximum_filter1d if fn == "max" else minimum_filter1d
    return f(a, size=2 * r + 1, axis=axis, mode="nearest")


def refine(m, threshold=0.5, softness=0.1, grow=0):
    """The matte -> alpha (0..1), with the refining controls (see the top of this file). Mirrors the window's
    preview (app.js: cutRefine)."""
    t = float(np.clip(threshold, 0.01, 0.99))
    s = float(np.clip(softness, 0.0, 1.0))
    lo, hi = t - s / 2, t + s / 2
    if hi - lo < 1e-3:
        a = (m >= t).astype(np.float32)
    else:
        a = np.clip((m - lo) / (hi - lo), 0, 1)
        a = a * a * (3 - 2 * a)   # smoothstep
    g = int(round(grow or 0))
    if g:
        fn = "max" if g > 0 else "min"
        a = _slide(_slide(a, abs(g), fn, 1), abs(g), fn, 0)
    return a.astype(np.float32)


def clean_bits(a, keep_frac=0.004):
    """drop small separate specks (less than 0.4 % of the picture), keep everything big"""
    import cv2
    solid = (a > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if n <= 2:
        return a
    min_area = keep_frac * a.size
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    drop = np.zeros(n, bool)
    for i in range(1, n):
        if i != biggest and stats[i, cv2.CC_STAT_AREA] < min_area:
            drop[i] = True
    if not drop.any():
        return a
    # take the dropped bits out together with their soft edges
    gone = drop[labels].astype(np.uint8)
    gone = cv2.dilate(gone, np.ones((5, 5), np.uint8))
    return a * (1 - gone)


def fix_edge_colours(rgb, a, radius=6):
    """In the see-through edge, use colours from inside the subject (spread outwards), so no fringe of the old
    background stays around it."""
    import cv2
    img = rgb.astype(np.float32)
    inner = (a > 0.95).astype(np.float32)
    if inner.sum() < 16:
        return rgb
    k = (radius * 2 + 1, radius * 2 + 1)
    num = cv2.GaussianBlur(img * inner[..., None], k, radius)
    den = cv2.GaussianBlur(inner, k, radius)[..., None]
    spread = np.where(den > 1e-3, num / np.maximum(den, 1e-3), img)
    w = np.clip((0.95 - a) / 0.95, 0, 1)[..., None]   # the more see-through, the more of the spread colour
    w = np.where(a[..., None] > 0.01, w, 0)
    return np.clip(img * (1 - w) + spread * w, 0, 255).astype(np.uint8)


def compose(image, alpha, background=None, fix_edges=True):
    """-> PIL RGBA (background None / "transparent") or RGB on a colour ("#rrggbb")."""
    rgb = np.asarray(image.convert("RGB"))
    if fix_edges:
        rgb = fix_edge_colours(rgb, alpha)
    a8 = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    if not background or background == "transparent":
        return Image.fromarray(np.dstack([rgb, a8]), "RGBA")
    c = str(background).lstrip("#")
    try:
        col = np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    except ValueError:
        col = np.array([255, 255, 255], np.float32)
    out = rgb.astype(np.float32) * alpha[..., None] + col * (1 - alpha[..., None])
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def selection_png(alpha, width, height, threshold=0.5):
    """the subject as a painted selection (pink where it is), the size of the painting area -> PNG bytes"""
    import io
    m = Image.fromarray((alpha >= threshold).astype(np.uint8) * 255, "L").resize((int(width), int(height)), Image.BILINEAR)
    on = np.asarray(m) >= 128
    rgba = np.zeros((on.shape[0], on.shape[1], 4), np.uint8)
    rgba[on] = (255, 64, 129, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()
