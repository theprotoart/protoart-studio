"""🩹 Remove (spot healing): paint over something and it's filled in from its surroundings, like Photoshop's
healing brush.

Two ways:
* "lama" (default): LaMa, a model made for exactly this (Apache-2.0, Samsung AI / advimman/lama, ONNX export by
  Carve). It sees the whole area around the painted part, so it continues textures, lines and backgrounds.
  ~208 MB, downloaded once from Hugging Face (a pinned revision; huggingface_hub checks the file), then offline.
  Runs on the CPU with ONNX Runtime (no image model needs to be loaded): a few seconds.
* "quick": OpenCV's Telea fill. No download, instant; fine for small spots, scratches and dust.

Only the painted area changes: the result is blended back into the untouched original.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

LAMA_REPO = "Carve/LaMa-ONNX"
LAMA_REVISION = "c3c0c9e468934d62e79c329e35d82dd09ff8c444"   # pinned: the files can't change under us
LAMA_FILE = "lama_fp32.onnx"
LAMA_SIZE = 512          # the export takes exactly 512 x 512
LAMA_MB = 208

_session = None


def lama_session(download=True):
    """The LaMa model, loaded once (ONNX Runtime, CPU)."""
    global _session
    if _session is None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        path = hf_hub_download(LAMA_REPO, LAMA_FILE, revision=LAMA_REVISION, local_files_only=not download)
        import os
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))  # all the performance cores
        _session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def lama_downloaded():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(LAMA_REPO, LAMA_FILE, revision=LAMA_REVISION, local_files_only=True)
        return True
    except Exception:
        return False


def _box(mask_np, pad_frac=0.6, min_side=LAMA_SIZE // 2):
    """The part of the picture LaMa looks at: the painted area plus plenty of surroundings (it needs context),
    square-ish, inside the picture."""
    ys, xs = np.nonzero(mask_np)
    h, w = mask_np.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    side = int(max(min_side, side * (1 + 2 * pad_frac)))
    side = min(side, max(w, h))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bx0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
    by0 = int(np.clip(cy - side // 2, 0, max(0, h - side)))
    return bx0, by0, min(w, bx0 + side), min(h, by0 + side)


def _run_lama(sess, crop_rgb, crop_mask):
    """crop_rgb: PIL RGB, crop_mask: PIL L (255 = fill). Returns a PIL RGB of the same size."""
    w, h = crop_rgb.size
    img = np.asarray(crop_rgb.resize((LAMA_SIZE, LAMA_SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
    m = np.asarray(crop_mask.resize((LAMA_SIZE, LAMA_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
    m = (m > 0.5).astype(np.float32)
    img_t = img.transpose(2, 0, 1)[None]
    m_t = m[None, None]
    inputs = sess.get_inputs()
    names = [i.name for i in inputs]
    mask_name = next((n for n in names if "mask" in n.lower()), names[-1])
    img_name = next((n for n in names if n != mask_name), names[0])
    out = sess.run(None, {img_name: img_t, mask_name: m_t})[0][0]
    out = out.transpose(1, 2, 0)
    if float(out.max()) <= 1.5:   # some exports give 0..1, others 0..255
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out).resize((w, h), Image.LANCZOS)


def _pil(mask_np):
    return Image.fromarray(mask_np.astype(np.uint8) * 255, "L")


def _parts(mask_np, join=24):
    """The painted area split into separate spots (spots closer than `join` px stay together), largest first."""
    import cv2
    near = cv2.dilate(mask_np.astype(np.uint8), np.ones((join, join), np.uint8))
    n, labels = cv2.connectedComponents(near)
    parts = [mask_np & (labels == i) for i in range(1, n)]
    parts = [p for p in parts if p.any()]
    return sorted(parts, key=lambda p: -int(p.sum())) or [mask_np]


def _thickness(mask_np):
    """How thick a painted spot is (its area over its longest side), in px."""
    ys, xs = np.nonzero(mask_np)
    return float(mask_np.sum()) / max(1, max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def _windows(mask_np, size=LAMA_SIZE, overlap=0.35):
    """Full-size LaMa windows along a long thin spot (with picture around it), overlapping so the fills join up."""
    h, w = mask_np.shape
    ys, xs = np.nonzero(mask_np)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    size = min(size, w, h)
    step = max(32, int(size * (1 - overlap)))

    def starts(a0, a1, full):
        lo = max(0, min(full - size, (a0 + a1) // 2 - size // 2)) if a1 - a0 <= size else max(0, a0 - size // 4)
        out, s = [], lo
        while True:
            s = min(s, full - size)
            out.append(s)
            if s + size >= min(full, a1 + size // 4) or s >= full - size:
                return out
            s += step

    return [(sx, sy, sx + size, sy + size) for sy in starts(y0, y1, h) for sx in starts(x0, x1, w)]


def _quick(image, mask):
    import cv2
    rgb = np.asarray(image.convert("RGB"))
    m = (np.asarray(mask) > 127).astype(np.uint8) * 255
    radius = 3 if m.sum() / 255 < 2000 else 7
    out = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def heal(image, mask, method="lama", session=None, feather=2):
    """image: PIL RGB; mask: PIL L the same size (white = remove). Returns the healed PIL RGB."""
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.NEAREST)
    mnp = np.asarray(mask) > 127
    if not mnp.any():
        raise ValueError("Paint over what you want to remove first.")
    # grow the painted area a little: the edge of an object usually has a halo that must go too
    grown = mask.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(5))
    if method == "quick":
        filled = _quick(image, grown)
    else:
        sess = session or lama_session()
        filled = image.copy()
        gnp = np.asarray(grown) > 127
        for part in _parts(gnp):  # each separate painted spot on its own, at the best size for it
            x0, y0, x1, y1 = _box(part)
            if (x1 - x0) <= LAMA_SIZE * 1.5 or _thickness(part) > LAMA_SIZE * 0.3:
                crop = _run_lama(sess, filled.crop((x0, y0, x1, y1)), _pil(part).crop((x0, y0, x1, y1)))
                filled.paste(Image.composite(crop, filled.crop((x0, y0, x1, y1)), _pil(part).crop((x0, y0, x1, y1))), (x0, y0))
            else:  # a long thin mark (a watermark, a line of text, a wire) on a big picture: worked along at full size
                for wx0, wy0, wx1, wy1 in _windows(part):
                    wm = _pil(part).crop((wx0, wy0, wx1, wy1))
                    if not wm.getbbox():
                        continue
                    crop = _run_lama(sess, filled.crop((wx0, wy0, wx1, wy1)), wm)
                    filled.paste(Image.composite(crop, filled.crop((wx0, wy0, wx1, wy1)), wm), (wx0, wy0))
    # only the painted area changes, with a soft edge so it doesn't show
    soft = grown.filter(ImageFilter.GaussianBlur(feather)) if feather else grown
    return Image.composite(filled, image, soft)
