"""Batch edit & export (Library → Select → Batch edit & export), in the spirit of PhotoMill.

The page sends one picture at a time (so it can show progress and stop at any point). For each one:
  * an Adjust preset or Look, if chosen, has already been applied by the page (on the Mac's graphics chip) and the
    result handed over as a temporary PNG in ~/.protoartstudio/adjust;
  * here it's resized, converted to sRGB, saved as PNG / JPEG / WebP / TIFF with the chosen quality and DPI,
    renamed from a pattern, and written to the chosen folder (never overwriting anything);
  * picture details (prompt, settings) are left out unless you choose to keep them: privacy first.
Pure PIL, unit-tested without a GPU.
"""
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

FORMATS = {"png": ("PNG", ".png"), "jpeg": ("JPEG", ".jpg"), "webp": ("WEBP", ".webp"), "tiff": ("TIFF", ".tif")}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
MAX_SIDE = 16384
TOKENS = ("name", "n", "date", "time", "model", "seed", "width", "height", "preset")


def target_size(w, h, resize):
    """resize: {"mode": "none"|"long"|"short"|"percent"|"fit"|"width"|"height", "value", "w", "h", "no_upscale"}."""
    r = resize or {}
    mode = r.get("mode") or "none"
    try:
        v = float(r.get("value") or 0)
    except (TypeError, ValueError):
        v = 0
    if mode == "long" and v > 0:
        s = v / max(w, h)
    elif mode == "short" and v > 0:
        s = v / min(w, h)
    elif mode == "width" and v > 0:
        s = v / w
    elif mode == "height" and v > 0:
        s = v / h
    elif mode == "percent" and v > 0:
        s = v / 100.0
    elif mode == "fit":
        try:
            bw, bh = float(r.get("w") or 0), float(r.get("h") or 0)
        except (TypeError, ValueError):
            bw = bh = 0
        s = min(bw / w if bw > 0 else 1e9, bh / h if bh > 0 else 1e9)
        if s >= 1e9:
            s = 1.0
    else:
        s = 1.0
    if r.get("no_upscale", True):
        s = min(s, 1.0)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    if max(nw, nh) > MAX_SIDE:
        k = MAX_SIDE / max(nw, nh)
        nw, nh = max(1, round(nw * k)), max(1, round(nh * k))
    return nw, nh


def _clean(text, limit=60):
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(text or "")).strip(" ._")
    return text[:limit]


def file_name(pattern, info):
    """Expand {name} {n} {date} {time} {model} {seed} {width} {height} {preset}; safe for any file system."""
    pattern = str(pattern or "{name}")[:200]

    def sub(m):
        key = m.group(1)
        if key == "n":
            return f"{int(info.get('n') or 1):03d}"
        return _clean(info.get(key, ""), 60)
    out = re.sub(r"\{(" + "|".join(TOKENS) + r")\}", sub, pattern)
    out = _clean(out, 150)
    return out or "picture"


def unique_path(folder, stem, ext):
    p = Path(folder) / f"{stem}{ext}"
    n = 2
    while p.exists():
        p = Path(folder) / f"{stem}-{n}{ext}"
        n += 1
    return p


def to_srgb(img):
    """Convert to sRGB if the picture carries another colour profile."""
    icc = img.info.get("icc_profile")
    if not icc:
        return img
    try:
        from PIL import ImageCms
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "srgb" in (ImageCms.getProfileDescription(src) or "").lower():
            return img
        mode = "RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB"
        return ImageCms.profileToProfile(img.convert(mode), src, ImageCms.createProfile("sRGB"), outputMode=mode)
    except Exception:
        return img


def _srgb_icc():
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def meta_text(settings):
    s = settings or {}
    loras = "".join(f" <lora:{l.get('name')}:{l.get('weight')}>" for l in (s.get("loras") or []) if isinstance(l, dict))
    return (f"{s.get('prompt', '')}{loras}\nNegative prompt: {s.get('negative_prompt', '')}\n"
            f"Steps: {s.get('steps')}, Sampler: {s.get('sampler')}, CFG scale: {s.get('cfg')}, "
            f"Seed: {s.get('seed')}, Size: {s.get('width')}x{s.get('height')}, Model: {s.get('model')}")


def export_one(src_path, dest_dir, opts, info=None, settings=None):
    """Save one picture. opts: {format, quality, resize, dpi, keep_meta, pattern}. info: values for the file-name
    pattern. settings: the picture's generation details (only written when keep_meta). -> dict about the file."""
    from PIL import Image, ImageOps
    fmt = str(opts.get("format") or "png").lower()
    if fmt == "same":  # keep each picture's own format
        fmt = {".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp", ".tif": "tiff", ".tiff": "tiff"}.get(
            (opts.get("_orig_ext") or Path(src_path).suffix).lower(), "png")
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format: {fmt}")
    pil_fmt, ext = FORMATS[fmt]
    dest = Path(dest_dir)
    if not dest.is_absolute():
        raise ValueError("Choose a folder to save to.")
    dest.mkdir(parents=True, exist_ok=True)
    import pictures
    with Image.open(pictures.prepare(src_path)) as im:  # any picture the app can open (HEIC, TIFF… via macOS)
        im.load()
        img = ImageOps.exif_transpose(im)
        img.info = dict(im.info)
    img = to_srgb(img)
    w, h = img.size
    nw, nh = target_size(w, h, opts.get("resize"))
    if (nw, nh) != (w, h):
        img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").resize((nw, nh), Image.Resampling.LANCZOS)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if fmt == "jpeg":
        if has_alpha:  # JPEG has no transparency: put it on white
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
    else:
        img = img.convert("RGBA" if has_alpha else "RGB")
    try:
        dpi = int(opts.get("dpi") or 72)
    except (TypeError, ValueError):
        dpi = 72
    dpi = max(1, min(dpi, 2400))
    quality = max(1, min(int(opts.get("quality") or 92), 100))
    kw = {"dpi": (dpi, dpi)}
    icc = _srgb_icc()
    if icc:
        kw["icc_profile"] = icc
    keep = bool(opts.get("keep_meta")) and settings
    text = None
    if keep:
        try:
            from engine import _shareable  # file paths on this Mac become file names
            shared = _shareable(settings)
        except Exception:
            shared = {k: v for k, v in settings.items() if not (isinstance(v, str) and v.startswith(("/", "~")))}
        text = meta_text(shared)
    if fmt == "jpeg":
        kw.update(quality=quality, optimize=True, progressive=True)
        if text:
            kw["comment"] = text.encode("utf-8")
    elif fmt == "webp":
        kw.update(quality=quality, method=4)
        kw.pop("dpi", None)
    elif fmt == "png":
        kw["optimize"] = False
        if text:
            from PIL.PngImagePlugin import PngInfo
            pi = PngInfo()
            pi.add_text("parameters", text)
            pi.add_text("beestudio", json.dumps(shared))
            kw["pnginfo"] = pi
    elif fmt == "tiff":
        kw["compression"] = "tiff_lzw"
        if text:
            kw["description"] = text
    # Press: the smallest file that still measures as looking the same (JPEG / PNG)
    press_strength = str(opts.get("compress") or "off").lower()
    detail = ""
    data = None
    if press_strength in ("gentle", "balanced", "strong") and fmt in ("jpeg", "png"):
        import press
        if fmt == "jpeg":
            jkw = {k: v for k, v in kw.items() if k in ("dpi", "icc_profile", "comment")}
            data, q = press.smallest_jpeg(img, press_strength, **jkw)
            detail = f"pressed · quality {q}"
        else:
            pkw = {k: v for k, v in kw.items() if k in ("dpi", "icc_profile", "pnginfo")}
            data, how = press.smallest_png(img, press_strength, **pkw)
            detail = f"pressed · {how}"
    stem = file_name(opts.get("pattern") or "{name}", info or {"name": Path(src_path).stem})
    out = unique_path(dest, stem, ext)
    tmp = out.with_name(out.name + ".part")
    try:
        if data is not None:
            tmp.write_bytes(data)
        else:
            img.save(tmp, pil_fmt, **kw)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"path": str(out), "name": out.name, "width": img.width, "height": img.height,
            "bytes": out.stat().st_size, "src_bytes": os.path.getsize(src_path), "kept_details": bool(keep),
            "format": fmt, "detail": detail}


def name_info(rec, src_path, n, preset=""):
    """Values for the file-name pattern from a Library record (or just a file)."""
    rec = rec or {}
    created = rec.get("created_at") or rec.get("created") or ""
    try:
        if isinstance(created, (int, float)):
            dt = datetime.fromtimestamp(created)
        elif created:
            dt = datetime.fromisoformat(str(created).replace("Z", ""))
        else:
            dt = datetime.fromtimestamp(os.path.getmtime(src_path))
    except (ValueError, OSError, OverflowError):
        dt = datetime.now()
    model = str(rec.get("model") or "")
    model = os.path.basename(model.split(":", 1)[-1]).rsplit(".", 1)[0] if model else ""
    return {"name": Path(rec.get("path") or src_path).stem, "n": n, "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H%M%S"), "model": model, "seed": rec.get("seed") if rec.get("seed") is not None else "",
            "width": rec.get("width") or "", "height": rec.get("height") or "", "preset": preset or ""}
