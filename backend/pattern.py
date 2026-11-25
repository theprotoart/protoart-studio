"""🔁 Seamless pattern: a picture that repeats with no visible join, like a fabric print, a wallpaper or a game texture.

How it works (the usual offset-and-repaint method, which works with any edit model):
1. the picture is shifted by half its width and height (it wraps round), so its four edges meet in a cross in the middle,
   and its new edges are lines from the middle of the old picture, which already join up perfectly when it repeats;
2. the cross is softened and the edit model repaints only that band, so the pattern carries on through it.
Nothing near the edges is touched in step 2, so the result tiles with no seam.
"""
from pathlib import Path

USES = {  # what the pattern is for -> how it's described to the model
    "fabric": "a seamless repeating textile print pattern of {what}: a flat 2D design seen straight on, the motifs spread "
              "evenly over the whole picture at a medium scale, clean printed colours",
    "wallpaper": "a seamless repeating wallpaper pattern of {what}: an elegant flat design seen straight on, evenly spaced "
                 "motifs over the whole picture, a harmonious colour palette",
    "game": "a seamless tileable surface texture of {what}, seen straight down from above: realistic material detail, flat "
            "even lighting with no shadows or highlights falling one way, the same amount of detail everywhere, no perspective",
}
EDGES = (" The pattern covers the whole frame from edge to edge, and the motifs carry on past the edges as on a tile cut from "
         "a bigger piece. No border, no frame, no vignette, no text, no watermark, no single centred subject.")


def describe(use, what):
    return USES.get(use, USES["fabric"]).format(what=(what or "").strip() or "small flowers and leaves") + "." + EDGES


def first_prompt(source, use, what):
    """Step 1 (from words or from a picture): the pattern itself, before the seams are fixed."""
    if source == "inspired":
        extra = f" featuring {what.strip()}" if (what or "").strip() else ""
        return ("Turn Picture 1 into " + USES.get(use, USES["fabric"]).format(what="the things, colours and style of Picture 1")
                + extra + ". Use Picture 1 only as inspiration: repeat its motifs many times in a pattern." + EDGES)
    return "Replace this entire blank picture with " + describe(use, what)


SEAM_PROMPT = ("This picture is a repeating pattern with a soft blurred band running across the middle and down the middle. "
               "Fill the blurred band with the same pattern so it carries on smoothly and naturally through it: the same "
               "motifs, colours, scale, lighting and style as the rest, as sharp and detailed as everywhere else. No straight "
               "lines, seams, borders or grid lines, and nothing new added.")


def offset_half(im):
    """The picture shifted by half its width and height, wrapping round (what was at the edges is now in the middle)."""
    w, h = im.size
    out = im.copy()
    dx, dy = w // 2, h // 2
    out.paste(im.crop((dx, dy, w, h)), (0, 0))
    out.paste(im.crop((0, dy, dx, h)), (w - dx, 0))
    out.paste(im.crop((dx, 0, w, dy)), (0, h - dy))
    out.paste(im.crop((0, 0, dx, dy)), (w - dx, h - dy))
    return out


def seam_mask(size, frac=0.14):
    """White cross (the band to repaint) on black, frac of the shorter side wide, well away from the edges."""
    from PIL import Image, ImageDraw
    w, h = size
    band = max(16, int(min(w, h) * frac))
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rectangle((w // 2 - band // 2, 0, w // 2 + band // 2, h), fill=255)
    d.rectangle((0, h // 2 - band // 2, w, h // 2 + band // 2), fill=255)
    return m


def prepare(path, out_dir, stamp="0"):
    """A picture -> (the shifted picture with its cross softened, the cross mask): the two files for the seam pass."""
    from PIL import Image, ImageFilter, ImageOps
    import pictures
    with Image.open(pictures.prepare(str(path))) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if max(w, h) > 2048:  # big enough for a tile; the edit works at about 1024 anyway
        k = 2048 / max(w, h)
        im = im.resize((max(64, int(w * k)), max(64, int(h * k))), Image.LANCZOS)
    shifted = offset_half(im)
    mask = seam_mask(shifted.size)
    # the smudge stays inside the band (the band's middle, blurred), so nothing outside it changes
    soft = seam_mask(shifted.size, 0.08).filter(ImageFilter.GaussianBlur(max(2, min(shifted.size) // 60)))
    blurred = shifted.filter(ImageFilter.GaussianBlur(max(4, min(shifted.size) // 40)))
    shifted = Image.composite(blurred, shifted, soft)  # the hard join smudged, so it's redrawn rather than copied
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pic, m = out_dir / f"pattern_shift_{stamp}.png", out_dir / f"pattern_mask_{stamp}.png"
    shifted.save(pic)
    mask.save(m)
    return {"path": str(pic), "mask": str(m), "width": shifted.width, "height": shifted.height}


def seam_score(im):
    """How visible the join is when the picture repeats: the jump across the wrapped edges compared with the picture's
    usual jump between neighbouring pixels (about 1 = no visible seam; much more = a seam)."""
    import numpy as np
    a = np.asarray(im.convert("L"), dtype=np.float32)
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2 + 1e-3
    wrap = (np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0, :] - a[-1, :]).mean()) / 2
    return float(wrap / inner)
