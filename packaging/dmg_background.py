"""Draws the DMG window's background: dark rack panel, an amber arrow from the app to Applications, and a line
of instructions, plus a light plate behind each icon's name (Finder draws the names in dark text, unreadable on the
dark panel otherwise). Writes <out>/bg.png (660×460) and <out>/bg@2x.png (1320×920).
Usage: python dmg_background.py <out dir>"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 660, 460
APP_X, APPS_X, ICON_Y = 170, 490, 160   # must match the icon positions in make_dmg.sh
README_X, README_Y = 330, 335           # "Read me first.txt", below the instructions
ICON = 96                               # icon size set in make_dmg.sh


def font(size, bold=False):
    for path in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(scale):
    w, h = W * scale, H * scale
    img = Image.new("RGB", (w, h), (24, 25, 28))
    d = ImageDraw.Draw(img)
    # brushed-metal stripes
    for y in range(0, h, 2 * scale):
        shade = 24 + (y * 7 // h) % 3
        d.line([(0, y), (w, y)], fill=(shade, shade + 1, shade + 4))
    # amber stripe at the top, like the app's rack
    d.rectangle([0, 0, w, 6 * scale], fill=(236, 144, 32))
    # title
    t = "PROTOART STUDIO"
    f = font(22 * scale, True)
    tw = d.textlength(t, font=f)
    d.text(((w - tw) / 2, 26 * scale), t, font=f, fill=(239, 234, 220))
    # arrow from the app to Applications (glow + body)
    y = ICON_Y * scale
    x0, x1 = (APP_X + 70) * scale, (APPS_X - 70) * scale
    glow = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(glow)
    gd.line([(x0, y), (x1 - 18 * scale, y)], fill=180, width=14 * scale)
    gd.polygon([(x1, y), (x1 - 30 * scale, y - 20 * scale), (x1 - 30 * scale, y + 20 * scale)], fill=180)
    glow = glow.filter(ImageFilter.GaussianBlur(8 * scale))
    img.paste((255, 160, 40), mask=glow.point(lambda v: v // 3))
    d.line([(x0, y), (x1 - 18 * scale, y)], fill=(255, 174, 43), width=6 * scale)
    d.polygon([(x1, y), (x1 - 26 * scale, y - 16 * scale), (x1 - 26 * scale, y + 16 * scale)], fill=(255, 174, 43))
    # light plates under the icon names (Finder writes them in dark text)
    def plate(cx, cy, pw=150):
        d.rounded_rectangle([(cx - pw / 2) * scale, (cy + ICON / 2 + 3) * scale, (cx + pw / 2) * scale, (cy + ICON / 2 + 25) * scale],
                            radius=5 * scale, fill=(222, 217, 205), outline=(120, 114, 102), width=max(1, scale))
    plate(APP_X, ICON_Y)
    plate(APPS_X, ICON_Y)
    plate(README_X, README_Y, 140)
    # instructions, between the icons' names and the read-me
    s = "Drag Protoart Studio onto Applications to install"
    f2 = font(15 * scale)
    sw = d.textlength(s, font=f2)
    d.text(((w - sw) / 2, (ICON_Y + ICON / 2 + 40) * scale), s, font=f2, fill=(220, 214, 200))
    s2 = 'First time: System Settings > Privacy & Security > Open Anyway  ·  see "Read me first"'
    f3 = font(11 * scale)
    sw2 = d.textlength(s2, font=f3)
    d.text(((w - sw2) / 2, (H - 30) * scale), s2, font=f3, fill=(150, 146, 138))
    return img


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    draw(1).save(os.path.join(out, "bg.png"))
    draw(2).save(os.path.join(out, "bg@2x.png"))


if __name__ == "__main__":
    main()
