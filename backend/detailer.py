"""Face detailer (like ADetailer): find each face, redraw it larger with the same model and prompt,
and blend it back. Faces come out sharper and in the image's own style.

Also: helpers for inpainting masks and outpainting (expanding the canvas).
"""
import numpy as np


class FaceFinder:
    """RetinaFace (facexlib, already used by Restore faces). Runs on the CPU."""

    def __init__(self, weights_dir):
        import torch
        from facexlib.detection import init_detection_model
        self.torch = torch
        self.net = init_detection_model("retinaface_resnet50", half=False, device=torch.device("cpu"),
                                        model_rootpath=str(weights_dir))

    def find(self, image, min_score=0.6, min_size=24):
        """-> list of face boxes (x1, y1, x2, y2), biggest first"""
        import cv2
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        with self.torch.no_grad():
            dets = self.net.detect_faces(bgr, conf_threshold=min_score)
        boxes = []
        for d in dets if dets is not None else []:
            x1, y1, x2, y2 = [float(v) for v in d[:4]]
            if min(x2 - x1, y2 - y1) >= min_size:
                boxes.append((x1, y1, x2, y2))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes


def face_prompts(text, n):
    """“prompt for face 1 || prompt for face 2 || …” (faces counted left to right) -> one prompt per face. Faces
    beyond the list use the last one; an empty part means that face keeps the main prompt."""
    parts = [p.strip() for p in (text or "").split("||")]
    if not parts:
        parts = [""]
    return [parts[min(k, len(parts) - 1)] for k in range(n)]


def face_crop(image, box, padding=1.8, work=512, grow=0.18, blur=1.0, hair=False):
    """Square region around a face, its size in the image, and the mask (ellipse) in work-size coords.
    grow: how far the mask reaches past the face (fraction of its width); blur: edge softness (1 = normal);
    hair: the mask also covers the hair above and beside the face."""
    from PIL import Image, ImageDraw, ImageFilter
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * padding
    side = min(side, max(image.width, image.height))
    left = int(max(0, min(image.width - side, cx - side / 2)))
    top = int(max(0, min(image.height - side, cy - side / 2)))
    side_i = int(min(side, image.width - left, image.height - top))
    region = (left, top, left + side_i, top + side_i)
    crop = image.crop(region).resize((work, work), Image.LANCZOS)
    s = work / side_i
    fx1, fy1, fx2, fy2 = (x1 - left) * s, (y1 - top) * s, (x2 - left) * s, (y2 - top) * s
    g = max(0.0, min(1.0, float(grow))) * (fx2 - fx1)
    mask = Image.new("L", (work, work), 0)
    if hair:  # the hair: a taller, wider shape reaching up over the head and down to the ears
        fw = fx2 - fx1
        ImageDraw.Draw(mask).ellipse([fx1 - g - 0.25 * fw, fy1 - g - 0.55 * fw, fx2 + g + 0.25 * fw, fy2 + g], fill=255)
    else:
        ImageDraw.Draw(mask).ellipse([fx1 - g, fy1 - g * 1.4, fx2 + g, fy2 + g], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(0.5, work / 48 * max(0.1, min(4.0, float(blur))))))
    return crop, region, mask


def paste_back(image, new_crop, region, mask):
    """Blend the redrawn face back using the (soft) mask."""
    from PIL import Image
    left, top, right, bottom = region
    size = (right - left, bottom - top)
    patch = new_crop.resize(size, Image.LANCZOS)
    m = mask.resize(size, Image.LANCZOS)
    out = image.copy()
    out.paste(patch, (left, top), m)
    return out


# ------------------------------------------------------------------ inpainting / outpainting
def decode_mask(data, size):
    """data: 'data:image/png;base64,...' or a file path. White (or opaque) = redraw. -> PIL 'L' at size."""
    import base64
    import io
    from PIL import Image
    if isinstance(data, str) and data.startswith("data:"):
        raw = base64.b64decode(data.split(",", 1)[1])
        im = Image.open(io.BytesIO(raw))
    else:
        im = Image.open(data)
    im.load()
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        m = im.convert("RGBA").split()[-1]  # painted strokes are opaque
    else:
        m = im.convert("L")
    m = m.resize(size, Image.LANCZOS)
    return m.point(lambda v: 255 if v > 32 else 0)


def soften(mask, blur):
    from PIL import ImageFilter
    if blur and blur > 0:
        return mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def outpaint_canvas(image, left=0, right=0, top=0, bottom=0, overlap=16):
    """Bigger canvas with the picture placed inside. The new area is pre-filled with stretched,
    blurred edge colours (gives the model something sensible to start from) plus a little noise.
    -> (canvas RGB, mask L: white = area to fill)"""
    from PIL import Image, ImageDraw, ImageFilter
    w, h = image.size
    W, H = w + left + right, h + top + bottom
    base = image.convert("RGB").resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(max(W, H) / 24))
    arr = np.asarray(base).astype(np.float32)
    rng = np.random.default_rng(0)
    arr = np.clip(arr + rng.normal(0, 18, arr.shape), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr)
    canvas.paste(image.convert("RGB"), (left, top))
    mask = Image.new("L", (W, H), 255)
    ov_l = overlap if left else 0
    ov_t = overlap if top else 0
    ov_r = overlap if right else 0
    ov_b = overlap if bottom else 0
    ImageDraw.Draw(mask).rectangle([left + ov_l, top + ov_t, left + w - 1 - ov_r, top + h - 1 - ov_b], fill=0)
    return canvas, mask.filter(ImageFilter.GaussianBlur(overlap / 2))
