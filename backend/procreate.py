"""Procreate (.procreate) files: open the finished picture, or each layer on its own.

A .procreate file is a zip: Document.archive (an Apple keyed-archive property list: canvas size, the layers with
their names, opacity, visibility, blend mode and clipping) and, for every layer and for the flattened "composite",
a folder of 256 x 256 tiles named "column~row.chunk", each LZO1X-compressed premultiplied RGBA. Nothing in the file is
ever changed: it's only read.
"""
from __future__ import annotations

import plistlib
import re
import zipfile

import numpy as np
from PIL import Image

MAX_SIDE = 16384


class ProcreateError(Exception):
    pass


def lzo1x_decompress(src: bytes, out_len: int = 0) -> bytes:
    """LZO1X decompression, following the reference decoder (minilzo's lzo1x_d), in plain Python. A C version is
    used when the `lzo` module happens to be installed."""
    try:
        import lzo  # python-lzo
        return lzo.decompress(src, False, out_len)
    except Exception:
        pass
    out = bytearray()
    ip = 0

    def copy(dist, length):
        start = len(out) - dist
        if start < 0:
            raise ProcreateError("damaged tile")
        if dist >= length:
            out.extend(out[start:start + length])
        else:  # overlapping: repeat the last `dist` bytes
            pattern = bytes(out[start:])
            while length > 0:
                take = min(length, dist)
                out.extend(pattern[:take])
                length -= take

    def run_length(t, base):
        nonlocal ip
        if t == 0:
            t = base
            while src[ip] == 0:
                t += 255
                ip += 1
            t += src[ip]
            ip += 1
        return t

    MATCH, MATCH_NEXT, LITERAL, FIRST = 0, 1, 2, 3
    step = LITERAL
    t = 0
    if src[0] > 17:
        t = src[0] - 17
        ip = 1
        if t < 4:
            step = MATCH_NEXT
        else:
            out.extend(src[ip:ip + t])
            ip += t
            step = FIRST
    while True:
        if step == LITERAL:  # a new instruction
            t = src[ip]
            ip += 1
            if t >= 16:
                step = MATCH
                continue
            t = run_length(t, 15)
            out.extend(src[ip:ip + t + 3])
            ip += t + 3
            step = FIRST
            continue
        if step == FIRST:  # right after a literal run
            t = src[ip]
            ip += 1
            if t >= 16:
                step = MATCH
                continue
            copy(1 + 0x0800 + (t >> 2) + (src[ip] << 2), 3)
            ip += 1
        elif step == MATCH:
            if t >= 64:
                copy(1 + ((t >> 2) & 7) + (src[ip] << 3), (t >> 5) + 1)
                ip += 1
            elif t >= 32:
                t = run_length(t & 31, 31)
                copy(1 + (src[ip] >> 2) + (src[ip + 1] << 6), t + 2)
                ip += 2
            elif t >= 16:
                far = (t & 8) << 11
                t = run_length(t & 7, 7)
                dist = far + (src[ip] >> 2) + (src[ip + 1] << 6)
                ip += 2
                if dist == 0:
                    break  # end of the stream
                copy(dist + 0x4000, t + 2)
            else:
                copy(1 + (t >> 2) + (src[ip] << 2), 2)
                ip += 1
        else:  # MATCH_NEXT: t (1..3) literals, then a match
            out.extend(src[ip:ip + t])
            ip += t
            t = src[ip]
            ip += 1
            step = MATCH
            continue
        # match done: the last 2 bits say how many literals follow (0: a new instruction)
        t = src[ip - 2] & 3
        step = MATCH_NEXT if t else LITERAL
    return bytes(out)


# --------------------------------------------------------------------------------------------- the document
BLEND_NAMES = {0: "normal", 1: "multiply", 2: "screen", 3: "add", 4: "lighten", 5: "exclusion", 6: "difference",
               7: "subtract", 8: "linear burn", 9: "color dodge", 10: "color burn", 11: "overlay", 12: "hard light",
               13: "color", 14: "luminosity", 15: "hue", 16: "saturation", 17: "soft light", 19: "darken",
               20: "hard mix", 21: "vivid light", 22: "linear light", 23: "pin light", 24: "lighter color",
               25: "darker color", 26: "divide"}


def _uid(objs, v):
    return objs[v.data] if isinstance(v, plistlib.UID) else v


def read_document(zf):
    """-> {"name", "width", "height", "tile", "orientation", "flip_h", "flip_v", "background", "composite", "layers"}
    layers: top first, each {"uuid", "name", "opacity", "hidden", "blend", "clipped"}"""
    try:
        arch = plistlib.loads(zf.read("Document.archive"))
    except (KeyError, plistlib.InvalidFileException, ValueError):
        raise ProcreateError("This .procreate file has no readable document inside.")
    objs = arch["$objects"]
    root = objs[arch["$top"]["root"].data]
    R = lambda v: _uid(objs, v)  # noqa: E731
    m = re.match(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", str(R(root.get("size"))))
    if not m:
        raise ProcreateError("Couldn't read the canvas size.")
    w, h = int(m.group(1)), int(m.group(2))
    if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE):
        raise ProcreateError(f"The canvas is {w} × {h}: too big to open here.")
    layers = []
    for v in (R(root.get("layers")) or {}).get("NS.objects", []):
        l = R(v)
        if not isinstance(l, dict):
            continue
        layers.append({"uuid": str(R(l.get("UUID"))), "name": str(R(l.get("name")) or "Layer"),
                       "opacity": float(l.get("opacity", 1.0)), "hidden": bool(l.get("hidden")),
                       "blend": int(l.get("extendedBlend") or l.get("blend") or 0), "clipped": bool(l.get("clipped"))})
    comp = R(root.get("composite"))
    bg = R(root.get("backgroundColor"))
    background = None
    if isinstance(bg, (bytes, bytearray)) and len(bg) >= 16:
        rgba = np.frombuffer(bytes(bg[:16]), "<f4")
        background = tuple(int(round(float(x) * 255)) for x in np.clip(rgba, 0, 1))
    name = R(root.get("name"))
    icc = None
    prof = R(root.get("colorProfile"))
    if isinstance(prof, dict):
        data = R(prof.get("SiColorProfileArchiveICCDataKey"))
        icc = bytes(data) if isinstance(data, (bytes, bytearray)) else None
    return {"icc": icc,"name": str(name).strip() if isinstance(name, str) and name.strip() else "Procreate",
            "width": w, "height": h, "tile": int(root.get("tileSize") or 256),
            "orientation": int(root.get("orientation") or 1),
            "flip_h": bool(root.get("flippedHorizontally")), "flip_v": bool(root.get("flippedVertically")),
            "background": background, "background_hidden": bool(root.get("backgroundHidden")),
            "composite": str(R(comp.get("UUID"))) if isinstance(comp, dict) else None, "layers": layers}


def layer_image(zf, doc, uuid):
    """One layer (or the composite) as a straight-alpha RGBA PIL image the size of the canvas."""
    W, H, T = doc["width"], doc["height"], doc["tile"]
    canvas = np.zeros((H, W, 4), np.uint8)
    prefix = uuid + "/"
    found = False
    for name in zf.namelist():
        if not (name.startswith(prefix) and name.endswith(".chunk")):
            continue
        m = re.match(r".*/(\d+)~(\d+)\.chunk$", name)
        if not m:
            continue
        col, row = int(m.group(1)), int(m.group(2))
        tw, th = min(T, W - col * T), min(T, H - row * T)
        if tw <= 0 or th <= 0:
            continue
        raw = lzo1x_decompress(zf.read(name), tw * th * 4)
        if len(raw) != tw * th * 4:
            raise ProcreateError("A layer tile is damaged.")
        tile = np.frombuffer(raw, np.uint8).reshape(th, tw, 4)
        # tiles are stored bottom row first (the iPad's graphics convention): rows count up from the bottom
        y1 = H - row * T
        canvas[y1 - th:y1, col * T:col * T + tw] = tile[::-1]
        found = True
    if not found:
        return None
    a = canvas[..., 3:4].astype(np.float32)
    rgb = np.where(a > 0, canvas[..., :3].astype(np.float32) * 255.0 / np.maximum(a, 1), 0)  # un-premultiply
    img = Image.fromarray(np.dstack([np.clip(rgb + 0.5, 0, 255).astype(np.uint8), canvas[..., 3]]), "RGBA")
    return _orient(img, doc)


def to_srgb(img, doc):
    """Procreate canvases are often in Display P3: convert to sRGB (what the rest of the app and most apps expect),
    so the colours look the same as in Procreate."""
    if not doc.get("icc"):
        return img
    try:
        import io
        from PIL import ImageCms
        src = ImageCms.ImageCmsProfile(io.BytesIO(doc["icc"]))
        dst = ImageCms.createProfile("sRGB")
        alpha = img.getchannel("A") if img.mode == "RGBA" else None
        rgb = ImageCms.profileToProfile(img.convert("RGB"), src, dst, renderingIntent=0, outputMode="RGB")
        if alpha is not None:
            rgb.putalpha(alpha)
        return rgb
    except Exception:
        return img


def _orient(img, doc):
    o = doc.get("orientation", 1)
    if o == 3:
        img = img.transpose(Image.ROTATE_90)
    elif o == 4:
        img = img.transpose(Image.ROTATE_270)
    elif o == 2:
        img = img.transpose(Image.ROTATE_180)
    if doc.get("flip_h"):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if doc.get("flip_v"):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def open_file(path):
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise ProcreateError("That isn't a Procreate file this app can read.")


def flattened(path):
    """The finished picture (Procreate's own flattened copy), on its background colour -> (PIL RGB, doc)."""
    zf = open_file(path)
    doc = read_document(zf)
    img = layer_image(zf, doc, doc["composite"]) if doc["composite"] else None
    if img is None:
        raise ProcreateError("This Procreate file has no flattened picture inside: open it in Procreate and save it again.")
    bg = doc["background"] if doc["background"] and not doc["background_hidden"] else (255, 255, 255, 255)
    base = Image.new("RGBA", img.size, tuple(bg[:3]) + (255,))
    base.alpha_composite(img)
    return to_srgb(base.convert("RGB"), doc), doc


def layers(path, include_hidden=False):
    """Every layer that has something on it, top first -> [(info, PIL RGBA in sRGB)], doc"""
    zf = open_file(path)
    doc = read_document(zf)
    out = []
    for info in doc["layers"]:
        if info["hidden"] and not include_hidden:
            continue
        img = layer_image(zf, doc, info["uuid"])
        if img is None or not img.getchannel("A").getbbox():
            continue
        out.append((info, to_srgb(img, doc)))
    return out, doc
