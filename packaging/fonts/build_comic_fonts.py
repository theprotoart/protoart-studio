"""Builds Protoart Studio's own comic lettering fonts (app/renderer/assets/fonts/protoart/*.ttf).

Original designs made for Protoart Studio: every letter is a hand-placed skeleton of pen strokes (below), and each
font draws those skeletons with a different pen:
  * Protoart Speech       dialogue lettering: an even round pen, a slight forward lean, a little hand wobble
  * Protoart Speech Bold  the same, a heavier pen (for emphasis and shouting)
  * Protoart Kaboom       sound effects: a heavy square pen (blocky strokes, sharp corners), a strong lean and
                          letters that jump up and down
  * Protoart Chatter      casual, bouncy cartoon lettering: a fat round pen, upright, each letter tilted a little
Lowercase keys give slightly smaller capitals (comic lettering is written in capitals), so a word never repeats
exactly the same shape. Licensed under the SIL Open Font License 1.1 (see OFL-protoart.txt).

Run:  python3 packaging/fonts/build_comic_fonts.py   (needs fontTools)
"""
import math
import random
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

OUT = Path(__file__).resolve().parents[2] / "app" / "renderer" / "assets" / "fonts" / "protoart"
CAP = 700


# ------------------------------------------------------------------ skeleton helpers (font units, y up)
def arc(cx, cy, rx, ry, a0, a1, n=None):
    """Points on an ellipse from angle a0 to a1 (degrees, counter-clockwise when a1 > a0)."""
    n = n or max(6, int(abs(a1 - a0) / 9))
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def ring(cx, cy, rx, ry):
    return arc(cx, cy, rx, ry, 0, 360, 40)


def dot(x, y):
    return [(x, y), (x + 0.5, y)]


# Each glyph: (advance width without side bearings, [strokes]); a stroke is a list of points
G = {
    "A": (520, [[(0, 0), (260, 700), (520, 0)], [(90, 240), (430, 240)]]),
    "B": (470, [[(0, 0), (0, 700), (230, 700)] + arc(230, 535, 190, 165, 90, -90)[1:] + [(0, 370)],
                [(0, 370), (250, 370)] + arc(250, 185, 210, 185, 90, -90)[1:] + [(0, 0)]]),
    "C": (560, [arc(300, 350, 290, 350, 45, 315)]),
    "D": (540, [[(0, 0), (0, 700), (200, 700)] + arc(200, 350, 330, 350, 90, -90)[1:] + [(0, 0)]]),
    "E": (460, [[(460, 700), (0, 700), (0, 0), (470, 0)], [(0, 360), (360, 360)]]),
    "F": (450, [[(460, 700), (0, 700), (0, 0)], [(0, 360), (360, 360)]]),
    "G": (590, [arc(300, 350, 290, 350, 42, 340), [(360, 320), (590, 320), (590, 70)]]),
    "H": (500, [[(0, 0), (0, 700)], [(500, 0), (500, 700)], [(0, 360), (500, 360)]]),
    "I": (300, [[(150, 0), (150, 700)], [(0, 700), (300, 700)], [(0, 0), (300, 0)]]),
    "J": (450, [[(170, 700), (450, 700)], [(400, 700), (400, 220)] + arc(215, 220, 185, 220, 0, -180)[1:]]),
    "K": (500, [[(0, 0), (0, 700)], [(480, 700), (0, 260)], [(160, 405), (510, 0)]]),
    "L": (430, [[(0, 700), (0, 0), (440, 0)]]),
    "M": (620, [[(0, 0), (30, 700), (310, 230), (590, 700), (620, 0)]]),
    "N": (520, [[(0, 0), (0, 700), (520, 0), (520, 700)]]),
    "O": (600, [ring(300, 350, 300, 350)]),
    "P": (460, [[(0, 0), (0, 700), (250, 700)] + arc(250, 515, 200, 185, 90, -90)[1:] + [(0, 330)]]),
    "Q": (610, [ring(300, 350, 300, 350), [(340, 190), (620, -70)]]),
    "R": (490, [[(0, 0), (0, 700), (250, 700)] + arc(250, 520, 200, 180, 90, -90)[1:] + [(0, 340)], [(210, 340), (500, 0)]]),
    "S": (500, [arc(250, 527, 230, 173, 15, 270) + arc(250, 175, 250, 175, 90, -200)[1:]]),
    "T": (560, [[(0, 700), (560, 700)], [(280, 700), (280, 0)]]),
    "U": (520, [[(0, 700), (0, 250)] + arc(260, 250, 260, 250, 180, 360)[1:] + [(520, 700)]]),
    "V": (540, [[(0, 700), (270, 0), (540, 700)]]),
    "W": (680, [[(0, 700), (160, 0), (340, 520), (520, 0), (680, 700)]]),
    "X": (520, [[(0, 700), (520, 0)], [(520, 700), (0, 0)]]),
    "Y": (540, [[(0, 700), (270, 340), (540, 700)], [(270, 340), (270, 0)]]),
    "Z": (500, [[(10, 700), (500, 700), (0, 0), (510, 0)]]),
    "0": (470, [ring(235, 350, 235, 350), [(110, 120), (360, 580)]]),
    "1": (380, [[(40, 540), (220, 700), (220, 0)], [(40, 0), (380, 0)]]),
    "2": (470, [arc(235, 480, 215, 220, 160, -35) + [(0, 0), (480, 0)]]),
    "3": (460, [arc(220, 527, 210, 173, 155, -90) + arc(220, 175, 240, 175, 90, -160)[1:]]),
    "4": (480, [[(370, 0), (370, 700), (0, 210), (490, 210)]]),
    "5": (470, [[(440, 700), (60, 700), (30, 400), (170, 430)] + arc(240, 225, 230, 225, 110, -155)[1:]]),
    "6": (480, [arc(250, 380, 230, 320, 68, 215), ring(240, 215, 225, 215)]),
    "7": (470, [[(0, 700), (470, 700), (150, 0)]]),
    "8": (470, [ring(235, 532, 190, 168), ring(235, 182, 235, 182)]),
    "9": (480, [ring(240, 485, 225, 215), arc(230, 320, 230, 320, 5, -112)]),
    ".": (60, [dot(30, 20)]),
    ",": (90, [[(70, 40), (30, -110)]]),
    "!": (80, [[(50, 700), (35, 220)], dot(30, 20)]),
    "?": (430, [arc(215, 520, 200, 180, 165, -40) + [(215, 330), (215, 220)], dot(215, 20)]),
    "'": (60, [[(40, 700), (25, 520)]]),
    "\"": (180, [[(40, 700), (25, 520)], [(160, 700), (145, 520)]]),
    "-": (300, [[(0, 300), (300, 300)]]),
    ":": (60, [dot(30, 20), dot(30, 410)]),
    ";": (90, [[(70, 40), (30, -110)], dot(60, 410)]),
    "(": (230, [arc(470, 330, 440, 470, 140, 220)]),
    ")": (230, [arc(-240, 330, 440, 470, 40, -40)]),
    "/": (380, [[(0, -80), (380, 780)]]),
    "+": (420, [[(0, 330), (420, 330)], [(210, 120), (210, 540)]]),
    "=": (420, [[(0, 230), (420, 230)], [(0, 440), (420, 440)]]),
    "*": (360, [[(180, 700), (180, 400)], [(40, 640), (320, 460)], [(320, 640), (40, 460)]]),
    "#": (480, [[(140, 0), (220, 700)], [(320, 0), (400, 700)], [(20, 230), (480, 230)], [(40, 470), (500, 470)]]),
    "%": (560, [ring(110, 560, 100, 130), [(40, 0), (520, 700)], ring(450, 140, 100, 130)]),
    "&": (560, [arc(200, 560, 140, 140, -40, 220) + [(470, 0)], [(80, 460)] + arc(240, 200, 220, 200, 150, 330)[1:] + [(540, 320)]]),
    "@": (700, [ring(350, 330, 150, 170), arc(350, 330, 340, 380, -20, 330), [(500, 160), (500, 500)]]),
    "$": (500, [arc(250, 527, 230, 173, 15, 270) + arc(250, 175, 250, 175, 90, -200)[1:], [(250, -80), (250, 780)]]),
    "~": (420, [arc(105, 320, 105, 60, 180, 0) + arc(315, 320, 105, 60, 180, 360)[1:]]),
    "<": (380, [[(380, 600), (0, 330), (380, 60)]]),
    ">": (380, [[(0, 600), (380, 330), (0, 60)]]),
    "_": (480, [[(0, -100), (480, -100)]]),
    "[": (200, [[(200, 760), (0, 760), (0, -100), (200, -100)]]),
    "]": (200, [[(0, 760), (200, 760), (200, -100), (0, -100)]]),
}
NAMES = {".": "period", ",": "comma", "!": "exclam", "?": "question", "'": "quotesingle", "\"": "quotedbl", "-": "hyphen",
         ":": "colon", ";": "semicolon", "(": "parenleft", ")": "parenright", "/": "slash", "+": "plus", "=": "equal",
         "*": "asterisk", "#": "numbersign", "%": "percent", "&": "ampersand", "@": "at", "$": "dollar", "~": "asciitilde",
         "<": "less", ">": "greater", "_": "underscore", "[": "bracketleft", "]": "bracketright",
         "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
         "9": "nine"}
EXTRA = {"’": "'", "‘": "'", "“": "\"", "”": "\"", "—": "-", "–": "-"}  # curly quotes, dashes


# ------------------------------------------------------------------ pens
def round_pen(r, seg=10):
    """Each stroke segment as a capsule (overlapping contours; the font draws them filled, non-zero)."""
    def capsule(a, b):
        (x0, y0), (x1, y1) = a, b
        ang = math.atan2(y1 - y0, x1 - x0)
        pts = []
        for i in range(seg + 1):  # half circle round b, then round a
            t = ang - math.pi / 2 + math.pi * i / seg
            pts.append((x1 + r * math.cos(t), y1 + r * math.sin(t)))
        for i in range(seg + 1):
            t = ang + math.pi / 2 + math.pi * i / seg
            pts.append((x0 + r * math.cos(t), y0 + r * math.sin(t)))
        return pts
    return capsule


def chisel_pen(w, angle=35):
    """A broad flat nib held at `angle`: thick and thin strokes with flat ends (a parallelogram per segment)."""
    nx, ny = w / 2 * math.cos(math.radians(angle)), w / 2 * math.sin(math.radians(angle))
    def quad(a, b):
        (x0, y0), (x1, y1) = a, b
        # a little round blob at the joints so corners don't show gaps
        return [(x0 - nx, y0 - ny), (x1 - nx, y1 - ny), (x1 + nx * 1.0, y1 + ny), (x0 + nx, y0 + ny)]
    return quad


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out
    lo, up = half(pts), half(reversed(pts))
    return lo[:-1] + up[:-1]


def block_pen(h):
    """A square nib (side 2h): heavy, blocky strokes with sharp corners, for sound effects."""
    def block(a, b):
        return hull([(x + dx, y + dy) for x, y in (a, b) for dx in (-h, h) for dy in (-h, h)])
    return block


STYLES = {
    "ProtoartSpeech-Regular": dict(family="Protoart Speech", style="Regular", pen=round_pen(44), slant=0.17, jitter=6, lower=0.93, weight=400),
    "ProtoartSpeech-Bold": dict(family="Protoart Speech", style="Bold", pen=round_pen(64), slant=0.17, jitter=6, lower=0.93, weight=700),
    "ProtoartKaboom-Regular": dict(family="Protoart Kaboom", style="Regular", pen=block_pen(88), slant=0.2, jitter=5, lower=0.9,
                                   bounce=90, tilt=8, wide=1.18, weight=900),
    "ProtoartChatter-Regular": dict(family="Protoart Chatter", style="Regular", pen=round_pen(70), slant=0.0, jitter=8, lower=0.86,
                                    bounce=60, tilt=7, wide=1.05, weight=600),
}


def union(polys, k=4):
    """The pen strokes merged into one clean outline (with holes), by drawing them at k pixels per font unit and
    tracing the edge: an outline pen or text-stroke then follows the letter's edge only, not every overlapping stroke.
    -> [(points, is_hole)]"""
    import cv2
    import numpy as np
    xs = [x for c in polys for x, _ in c]
    ys = [y for c in polys for _, y in c]
    x0, y0 = min(xs) - 8, min(ys) - 8
    W, H = int((max(xs) - x0 + 8) * k), int((max(ys) - y0 + 8) * k)
    img = np.zeros((H, W), np.uint8)
    for c in polys:
        pts = np.array([[(x - x0) * k, H - (y - y0) * k] for x, y in c], np.int32)
        cv2.fillPoly(img, [pts], 255, lineType=cv2.LINE_8)
    found, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for i, c in enumerate(found):
        c = cv2.approxPolyDP(c, 0.9 * k, True)
        if len(c) < 3:
            continue
        pts = [(round(p[0][0] / k + x0), round((H - p[0][1]) / k + y0)) for p in c]
        pts = [p for j, p in enumerate(pts) if j == 0 or p != pts[j - 1]]
        if len(pts) >= 3 and abs(area(pts)) > 30:
            out.append((pts, hier[0][i][3] >= 0))  # a parent means it's a hole
    return out


def area(c):
    return sum(c[i][0] * c[(i + 1) % len(c)][1] - c[(i + 1) % len(c)][0] * c[i][1] for i in range(len(c))) / 2


def build(name, st):
    rnd = random.Random(name)
    fb = FontBuilder(1000, isTTF=True)
    keys = list(G) + [k.lower() for k in G if k.isalpha()]
    order = [".notdef", "space"] + [glyph_name(k) for k in keys] + [f"uni{ord(k):04X}" for k in EXTRA]
    fb.setupGlyphOrder(order)
    cmap = {32: "space", 160: "space"}
    for k in keys:
        cmap[ord(k)] = glyph_name(k)
    for k in EXTRA:
        cmap[ord(k)] = f"uni{ord(k):04X}"
    fb.setupCharacterMap(cmap)
    glyphs, metrics = {}, {}
    pen0 = TTGlyphPen(None)
    glyphs[".notdef"] = pen0.glyph()
    metrics[".notdef"] = (500, 0)
    glyphs["space"] = TTGlyphPen(None).glyph()
    metrics["space"] = (int(300 * st.get("wide", 1)), 0)
    pad = 70
    for k in keys + list(EXTRA):
        src = EXTRA.get(k, k)
        lower = src.isalpha() and src.islower()
        w, strokes = G[src.upper()]
        sx = st.get("wide", 1) * (0.95 if lower else 1)
        sy = st["lower"] if lower else 1.0
        bounce = rnd.uniform(-1, 1) * st.get("bounce", 0) if src.isalnum() else 0
        tilt = math.radians(rnd.uniform(-1, 1) * st.get("tilt", 0)) if src.isalnum() else 0
        cx, cy = w * sx / 2, CAP * sy / 2
        def tf(p):
            x, y = p[0] * sx, p[1] * sy
            x += rnd.uniform(-1, 1) * st["jitter"]
            y += rnd.uniform(-1, 1) * st["jitter"]
            # tilt round the letter's middle, then lean forward and bounce
            dx, dy = x - cx, y - cy
            x, y = cx + dx * math.cos(tilt) - dy * math.sin(tilt), cy + dx * math.sin(tilt) + dy * math.cos(tilt)
            return (x + y * st["slant"], y + bounce)
        pen = TTGlyphPen(None)
        contours = []
        for s in strokes:
            pts = [tf(p) for p in s]
            for a, b in zip(pts, pts[1:]):
                contours.append(st["pen"](a, b))
            if st.get("extra_round"):  # round joints for the chisel pen
                for p in pts:
                    contours.append(st["extra_round"](p, (p[0] + 0.5, p[1])))
        minx = min(x for c in contours for x, _ in c)
        shift = pad - minx
        maxx = max(x for c in contours for x, _ in c) + shift
        for c, hole in union([[(x + shift, y) for x, y in c] for c in contours]):
            if (area(c) > 0) != hole:  # TrueType: outer outlines clockwise, holes counter-clockwise
                c = c[::-1]
            pen.moveTo(c[0])
            for p in c[1:]:
                pen.lineTo(p)
            pen.closePath()
        g = pen.glyph()
        gname = f"uni{ord(k):04X}" if k in EXTRA else glyph_name(k)
        glyphs[gname] = g
        metrics[gname] = (int(maxx + pad), int(shift + 0))
    fb.setupGlyf(glyphs)
    for gname, g in glyphs.items():  # left side bearing = the glyph's real xMin
        g.recalcBounds(fb.font["glyf"])
        metrics[gname] = (metrics[gname][0], getattr(g, "xMin", 0))
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=950, descent=-260)
    full = st["family"] + ("" if st["style"] == "Regular" else " " + st["style"])
    fb.setupNameTable({
        "familyName": st["family"], "styleName": st["style"], "uniqueFontIdentifier": f"Protoart Studio: {full}",
        "fullName": full, "psName": name, "version": "Version 1.000",
        "copyright": "Copyright 2026 Protoart Studio. Original design.",
        "licenseDescription": "This Font Software is licensed under the SIL Open Font License, Version 1.1.",
        "licenseInfoURL": "https://openfontlicense.org",
    })
    fb.setupOS2(sTypoAscender=950, sTypoDescender=-260, sTypoLineGap=0, usWinAscent=1000, usWinDescent=300,
                usWeightClass=st["weight"], fsSelection=(0x20 if st["style"] == "Bold" else 0x40), achVendID="PRTO",
                sxHeight=int(CAP * st["lower"]), sCapHeight=CAP)
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)
    if st["style"] == "Bold":
        fb.font["head"].macStyle = 1
    OUT.mkdir(parents=True, exist_ok=True)
    fb.save(OUT / f"{name}.ttf")
    return OUT / f"{name}.ttf"


def glyph_name(k):
    if k.isalpha():
        return k if k.isupper() else k  # "A"…"Z", "a"…"z"
    return NAMES[k]


if __name__ == "__main__":
    for n, st in STYLES.items():
        p = build(n, st)
        print(p.name, p.stat().st_size, "bytes")
