"""🧩 Combine and 🧅 Peel layers: the drawing parts (no model here).

- match_light: a cut-out picks up the scene's light and colour (its brightness, contrast and colour cast become
  those of the scene around where it goes), so it doesn't look pasted on.
- place: a cut-out (or a whole picture) put into a box on the scene, standing on the box's bottom edge.
- combine_prompt: the words FLUX.2 Klein gets for "add this character to this scene".
- peel: a painted or selected part lifted off the picture as its own see-through layer.
- write_psd: layers saved as a layered Photoshop file (opens in Photoshop, Procreate, Affinity, Krita, GIMP, Photopea).
"""
from __future__ import annotations

import struct

import numpy as np
from PIL import Image, ImageFilter


# ------------------------------------------------------------------------------------------- light matching
def _lab(rgb):
    import cv2
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _rgb(lab):
    import cv2
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def match_light(subject, scene, strength=0.7, region=None):
    """subject: PIL RGBA (the cut-out); scene: PIL RGB. region: (x0, y0, x1, y1) of the scene to take the light
    from (around where the subject goes); None = the whole scene. strength 0..1.
    -> RGBA with the subject's light and colour moved towards the scene's (alpha unchanged)."""
    s = float(np.clip(strength, 0, 1))
    sub = np.asarray(subject.convert("RGBA"))
    a = sub[..., 3].astype(np.float32) / 255.0
    if s <= 0 or a.max() <= 0:
        return subject.convert("RGBA")
    sc = np.asarray(scene.convert("RGB"))
    if region:
        x0, y0, x1, y1 = [int(v) for v in region]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(sc.shape[1], x1), min(sc.shape[0], y1)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            sc = sc[y0:y1, x0:x1]
    L = _lab(sub[..., :3])
    S = _lab(sc).reshape(-1, 3)
    w = a.reshape(-1)
    keep = w > 0.5
    if keep.sum() < 16:
        return subject.convert("RGBA")
    px = L.reshape(-1, 3)[keep]
    mu_s, sd_s = px.mean(0), px.std(0) + 1e-3
    mu_t, sd_t = S.mean(0), S.std(0) + 1e-3
    out = L.copy()
    # brightness: move the mean part of the way and the contrast gently (a character keeps its own shading)
    gain_L = float(np.clip(sd_t[0] / sd_s[0], 0.7, 1.4))
    out[..., 0] = (L[..., 0] - mu_s[0]) * (1 + (gain_L - 1) * s * 0.6) + mu_s[0] + (mu_t[0] - mu_s[0]) * s * 0.55
    # colour: the scene's cast (its mean a/b) carried over, the subject's own colours kept
    for c in (1, 2):
        out[..., c] = L[..., c] + (mu_t[c] - mu_s[c]) * s * 0.6
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    rgb = _rgb(out)
    return Image.fromarray(np.dstack([rgb, sub[..., 3]]), "RGBA")


# ------------------------------------------------------------------------------------------- placing
def fit_box(src_size, box, scene_size):
    """box: normalised {x, y, w, h} (0..1 of the scene). The picture fitted inside it, keeping its shape, centred
    left-right and standing on the box's bottom edge -> (left, top, width, height) in scene pixels."""
    W, H = scene_size
    bx, by = float(box.get("x", 0)) * W, float(box.get("y", 0)) * H
    bw, bh = max(1.0, float(box.get("w", 0.3)) * W), max(1.0, float(box.get("h", 0.5)) * H)
    sw, sh = src_size
    k = min(bw / sw, bh / sh)
    w, h = max(1, round(sw * k)), max(1, round(sh * k))
    return round(bx + (bw - w) / 2), round(by + bh - h), w, h


def place(scene, pieces):
    """scene: PIL RGB; pieces: [(PIL RGBA, box)] in order (the last one in front).
    -> (the composite RGB, the area they cover as an L mask)"""
    out = scene.convert("RGBA")
    cover = Image.new("L", scene.size, 0)
    for img, box in pieces:
        left, top, w, h = fit_box(img.size, box, scene.size)
        piece = img.convert("RGBA").resize((w, h), Image.LANCZOS)
        layer = Image.new("RGBA", scene.size, (0, 0, 0, 0))
        layer.paste(piece, (left, top))
        out = Image.alpha_composite(out, layer)
        cover = Image.fromarray(np.maximum(np.asarray(cover), np.asarray(layer.getchannel("A"))))
    return out.convert("RGB"), cover


def blend_area(cover, grow_frac=0.06):
    """Where Klein may redraw when blending placed pieces in: what they cover plus room for their shadows and
    edges (more below them, where a shadow falls), soft-edged. -> L mask"""
    W, H = cover.size
    side = min(W, H)
    g = max(3, int(side * grow_frac) // 2 * 2 + 1)
    m = cover.point(lambda v: 255 if v > 24 else 0).filter(ImageFilter.MaxFilter(min(g, 61)))
    if g > 61:  # MaxFilter is slow when big: grow the rest by blurring and thresholding
        m = m.filter(ImageFilter.GaussianBlur(g / 4)).point(lambda v: 255 if v > 20 else 0)
    # a little extra under each piece for a contact shadow
    a = np.asarray(m, np.uint8)
    shift = max(1, int(side * 0.03))
    below = np.zeros_like(a)
    below[shift:] = a[:-shift]
    m = Image.fromarray(np.maximum(a, below))
    return m.filter(ImageFilter.GaussianBlur(max(2, side // 160)))


def subject_bbox(alpha, pad=0.02):
    """the subject's box in a float alpha (h, w), with a small margin -> (x0, y0, x1, y1) or None"""
    ys, xs = np.nonzero(alpha > 0.05)
    if not len(xs):
        return None
    h, w = alpha.shape
    p = int(max(h, w) * pad)
    return max(0, xs.min() - p), max(0, ys.min() - p), min(w, xs.max() + 1 + p), min(h, ys.max() + 1 + p)


def on_white(rgba):
    """a cut-out on a plain white background (how a reference picture reads most clearly to Klein)"""
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba.convert("RGBA")).convert("RGB")


def combine_prompt(user, count=1, placed=False):
    """The words for FLUX.2 Klein. Picture 1 is the scene; pictures 2.. are the characters or things to add."""
    user = (user or "").strip().rstrip(".")
    user = user[:1].upper() + user[1:]
    if count <= 1:
        who, refs = "the character", "picture 2"
    else:
        who, refs = "the characters", " and ".join(f"picture {i + 2}" for i in range(count))
    keep = (f"Keep {who}' faces, hair, bodies and clothes exactly as in {refs}." if count > 1
            else f"Keep {who}'s face, hair, body and clothes exactly as in {refs}.")
    light = ("Match the scene's lighting, shadows, colours and perspective, with natural shadows where "
             f"{who} touch the ground." if count > 1 else
             "Match the scene's lighting, shadows, colours and perspective, with a natural shadow where "
             f"{who} touches the ground.")
    if placed:
        lead = (f"Blend {who} already placed in picture 1 naturally into the scene, at the same place, size and pose; "
                "fix the edges so nothing looks pasted on.")
        parts = [lead, user + "." if user else "", keep, light, "Keep everything else in the scene the same."]
    else:
        parts = [f"Add {who} from {refs} into the scene in picture 1.", user + "." if user else "", keep, light,
                 "Keep the scene, camera angle and framing of picture 1 the same."]
    return " ".join(p for p in parts if p)


# ------------------------------------------------------------------------------------------- peeling
def peel(image, mask, feather=1.5, matte=None):
    """image: PIL RGB; mask: L (white = the part to lift off). matte: optional float (h, w) 0..1 from Cut out to
    tidy the edge inside the painted area. -> (the layer as RGBA, the hole as an L mask for filling in)"""
    m = np.asarray(mask.convert("L"), np.float32) / 255.0
    if matte is not None and matte.shape == m.shape:
        grown = np.asarray(mask.convert("L").filter(ImageFilter.MaxFilter(5)), np.float32) / 255.0
        m = np.minimum(grown, matte)
    if feather > 0:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather)),
                       np.float32) / 255.0
    if (m > 0.02).sum() == 0:
        raise ValueError("Paint over or select the part to peel off first.")
    rgb = np.asarray(image.convert("RGB"))
    layer = Image.fromarray(np.dstack([rgb, (m * 255 + 0.5).astype(np.uint8)]), "RGBA")
    # the hole to fill: all of it, a little past its edge so no outline is left behind
    side = max(image.size)
    g = max(3, (round(side / 160) // 2) * 2 + 1)
    hole = Image.fromarray(((m > 0.04) * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(min(g, 31)))
    return layer, hole


# ------------------------------------------------------------------------------------------- Photoshop file
def _pascal(name, pad=4):
    raw = name.encode("latin-1", "replace")[:255]
    b = bytes([len(raw)]) + raw
    while len(b) % pad:
        b += b"\0"
    return b


def _luni(name):
    u = name.encode("utf-16-be")
    data = struct.pack(">I", len(u) // 2) + u
    if len(data) % 2:
        data += b"\0"
    return b"8BIM" + b"luni" + struct.pack(">I", len(data)) + data


def write_psd(path, size, layers, composite=None):
    """A layered .psd (8-bit RGB). size: (W, H); layers: [(name, PIL RGBA canvas-sized, visible)] bottom first;
    composite: the flattened picture (made from the layers when not given). Uncompressed: big but always readable."""
    W, H = size
    if not layers:
        raise ValueError("No layers to save.")
    recs, chans = [], []
    for name, img, visible in layers:
        img = img.convert("RGBA")
        if img.size != (W, H):
            raise ValueError("Every layer must be the size of the picture.")
        box = img.getchannel("A").getbbox() or (0, 0, 1, 1)
        l, t, r, b = box
        a = np.asarray(img.crop(box))
        data = [(-1, a[..., 3]), (0, a[..., 0]), (1, a[..., 1]), (2, a[..., 2])]
        rec = struct.pack(">iiiiH", t, l, b, r, 4)
        for cid, ch in data:
            rec += struct.pack(">hI", cid, 2 + ch.size)
        flags = 0 if visible else 2
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(name or "Layer") + _luni(name or "Layer")
        rec += b"8BIMnorm" + struct.pack(">BBBB", 255, 0, flags, 0) + struct.pack(">I", len(extra)) + extra
        recs.append(rec)
        chans.append(b"".join(struct.pack(">H", 0) + np.ascontiguousarray(ch).tobytes() for _, ch in data))
    info = struct.pack(">h", len(layers)) + b"".join(recs) + b"".join(chans)
    if len(info) % 2:
        info += b"\0"
    layer_info = struct.pack(">I", len(info)) + info
    lmi = layer_info + struct.pack(">I", 0)  # no global layer mask
    if composite is None:
        comp = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        for _, img, visible in layers:
            if visible:
                comp = Image.alpha_composite(comp, img.convert("RGBA"))
        composite = comp
    c = np.asarray(composite.convert("RGB"))
    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, H, W, 8, 3))
        f.write(struct.pack(">I", 0))  # colour mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(lmi)) + lmi)
        f.write(struct.pack(">H", 0))
        for i in range(3):
            f.write(np.ascontiguousarray(c[..., i]).tobytes())
    return path
