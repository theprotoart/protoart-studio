"""Comparison grid: the same picture made with different settings, laid out side by side with labels.

Axes the app can vary: cfg, steps, sampler, seed, clip_skip, lora_weight (of one LoRA), lora (which LoRA),
strength (image to image). A job has an X axis and optionally a Y axis.
"""
AXES = {
    "cfg": "Guidance (CFG)",
    "steps": "Steps",
    "sampler": "Sampler",
    "seed": "Seed",
    "clip_skip": "CLIP skip",
    "lora_weight": "LoRA weight",
    "lora": "LoRA",
    "strength": "Change strength",
}
MAX_CELLS = 36


def parse_values(axis, raw):
    """'5, 7, 9' / '4-10 (+2)' / list -> list of values for the axis."""
    if isinstance(raw, list):
        items = raw
    else:
        text = str(raw or "").strip()
        import re
        m = re.match(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*\(\s*\+?\s*([\d.]+)\s*\)\s*$", text)
        if m and axis not in ("sampler", "lora"):
            a, b, step = float(m.group(1)), float(m.group(2)), float(m.group(3))
            items, v = [], a
            while v <= b + 1e-9 and len(items) < MAX_CELLS:
                items.append(round(v, 4))
                v += step if step > 0 else 1
        else:
            items = [x.strip() for x in text.split(",") if x.strip()]
    out = []
    for v in items:
        if axis in ("steps", "seed", "clip_skip"):
            out.append(int(float(v)))
        elif axis in ("cfg", "lora_weight", "strength"):
            out.append(float(v))
        else:
            out.append(str(v))
    return out


def apply(args, axis, value, lora_name=None):
    """args for one cell (a copy)."""
    a = dict(args)
    a["loras"] = [dict(l) for l in (args.get("loras") or [])]
    if axis == "lora_weight":
        for l in a["loras"]:
            if lora_name is None or l["name"] == lora_name:
                l["weight"] = value
    elif axis == "lora":
        a["loras"] = [l for l in a["loras"] if l["name"] != lora_name] + [{"name": value, "weight": 1.0}] \
            if value.lower() not in ("none", "-") else [l for l in a["loras"] if l["name"] != lora_name]
    else:
        a[axis] = value
    return a


def label(axis, value):
    if axis == "cfg":
        return f"CFG {value:g}"
    if axis == "steps":
        return f"{value} steps"
    if axis == "seed":
        return f"seed {value}"
    if axis == "clip_skip":
        return "CLIP skip off" if value == 1 else f"CLIP skip {value}"
    if axis == "lora_weight":
        return f"weight {value:g}"
    if axis == "strength":
        return f"strength {value:g}"
    return str(value)


def compose(images, xs, ys, x_axis, y_axis, title=""):
    """images: dict (xi, yi) -> PIL image (or None for a failed cell). Returns one PIL image."""
    from PIL import Image, ImageDraw, ImageFont
    sample = next((im for im in images.values() if im is not None), None)
    if sample is None:
        raise ValueError("no images")
    cw, ch = sample.size
    scale = min(1.0, 512 / max(cw, ch)) if len(xs) * len(ys) > 4 else 1.0
    cw, ch = int(cw * scale), int(ch * scale)
    font_size = max(14, cw // 18)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.load_default(size=font_size)
        except Exception:
            font = ImageFont.load_default()
    head = font_size * 2 + 8
    side = 0
    if y_axis:
        side = max(int(font.getlength(label(y_axis, y))) for y in ys) + 24
    top = head + (font_size + 12 if title else 0)
    W, H = side + cw * len(xs), top + ch * len(ys)
    out = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(out)
    if title:
        d.text((10, 6), title[:160], fill=(200, 200, 205), font=font)
    for xi, x in enumerate(xs):
        t = label(x_axis, x)
        tw = font.getlength(t)
        d.text((side + xi * cw + (cw - tw) / 2, top - head + (head - font_size) / 2), t, fill=(240, 240, 240), font=font)
    for yi, y in enumerate(ys):
        if y_axis:
            t = label(y_axis, y)
            d.text((10, top + yi * ch + (ch - font_size) / 2), t, fill=(240, 240, 240), font=font)
        for xi in range(len(xs)):
            im = images.get((xi, yi))
            x0, y0 = side + xi * cw, top + yi * ch
            if im is None:
                d.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(90, 40, 40), width=2)
                d.text((x0 + 10, y0 + 10), "failed", fill=(220, 120, 120), font=font)
            else:
                out.paste(im.convert("RGB").resize((cw, ch)), (x0, y0))
    return out
