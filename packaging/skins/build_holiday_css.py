#!/usr/bin/env python3
"""Builds app/renderer/holiday.css: the four holiday looks (Halloween, Thanksgiving, Christmas, New Year's).

All the art is drawn here as small SVGs (original, made for Protoart Studio): the stickers on the faceplate, the
themed lamps (jack-o'-lanterns whose eyes light up, Christmas bulbs, autumn leaves, stars) and the touches on the
launch screen. Run it again after changing a drawing:  python3 packaging/skins/build_holiday_css.py
"""
from pathlib import Path
from urllib.parse import quote

OUT = Path(__file__).resolve().parents[2] / "app" / "renderer" / "holiday.css"


def uri(svg):
    svg = " ".join(svg.split())
    return 'url("data:image/svg+xml,' + quote(svg, safe=" /=:;,'()-.") + '")'


def S(body, vb="0 0 64 64"):
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{vb}'>{body}</svg>"


# ------------------------------------------------------------------ lamps (drawn on a 16 × 16 grid)
def pumpkin_lamp(body, rib, eyes, stem="#3d6b1f"):
    return S(f"""
<path d='M8 3.2c.2-1.2.9-2 1.9-2.3' stroke='{stem}' stroke-width='1.5' fill='none' stroke-linecap='round'/>
<ellipse cx='5' cy='9.5' rx='3.8' ry='5' fill='{body}'/><ellipse cx='11' cy='9.5' rx='3.8' ry='5' fill='{body}'/>
<ellipse cx='8' cy='9.5' rx='3.6' ry='5.3' fill='{rib}'/>
<path d='M4.3 8.4l1.6-1.7 1.1 1.9zM11.7 8.4l-1.6-1.7-1.1 1.9z' fill='{eyes}'/>
<path d='M4.6 11.2h6.8l-.9 1.7-1-.8-1 .9-1-.9-1 .8z' fill='{eyes}'/>""", "0 0 16 16")


def bulb_lamp(glass, hi):
    return S(f"""
<rect x='6' y='.6' width='4' height='3.2' rx='.6' fill='#7c8288'/><rect x='6' y='1.5' width='4' height='.5' fill='#54595e'/>
<path d='M8 3.4c3.2 1.8 4.4 5.2 4.1 7.6-.3 2.6-2.1 4.3-4.1 4.3s-3.8-1.7-4.1-4.3C3.6 8.6 4.8 5.2 8 3.4z' fill='{glass}'/>
<ellipse cx='6.5' cy='8' rx='1' ry='2' fill='{hi}' opacity='.75'/>""", "0 0 16 16")


def leaf_lamp(c, vein):
    return S(f"""
<path d='M8 .8l1.4 3 2.4-1-.6 3.2 3.3-.5-1.6 2.8 2.2 1.4-3.7 1.4.6 1.8-3.2-.7L8 15.2 7.2 12.2l-3.2.7.6-1.8L.9 9.7l2.2-1.4-1.6-2.8 3.3.5-.6-3.2 2.4 1z' fill='{c}'/>
<path d='M8 4v11M8 8l-3 -2M8 8l3-2M8 10.5l-2.5 0M8 10.5l2.5 0' stroke='{vein}' stroke-width='.6' fill='none'/>""", "0 0 16 16")


def star_lamp(c, core):
    return S(f"""
<path d='M8 .7l2.1 4.6 5 .5-3.8 3.4 1.1 5L8 11.6l-4.4 2.6 1.1-5L.9 5.8l5-.5z' fill='{c}'/>
<path d='M8 4.2l1 2.3 2.5.2-1.9 1.7.6 2.5L8 9.6l-2.2 1.3.6-2.5-1.9-1.7 2.5-.2z' fill='{core}'/>""", "0 0 16 16")


LAMPS = {
    "halloween": {
        "off": pumpkin_lamp("#6a3a12", "#7a4416", "#2a1406", "#2e4a18"),
        "on": pumpkin_lamp("#ff8a1a", "#ff9d33", "#fff27a"),
        "green": pumpkin_lamp("#ff8a1a", "#ff9d33", "#9dff5a"),
        "red": pumpkin_lamp("#ff8a1a", "#ff9d33", "#ff3b2e"),
        "glow": "255, 170, 40",
    },
    "thanksgiving": {
        "off": leaf_lamp("#5a3a1c", "#3c2610"),
        "on": leaf_lamp("#ffb22e", "#a8580a"),
        "green": leaf_lamp("#9cc23a", "#4f6b12"),
        "red": leaf_lamp("#e2452a", "#7a1a0a"),
        "glow": "255, 176, 50",
    },
    "christmas": {
        "off": bulb_lamp("#5a4a22", "#8a7a52"),
        "on": bulb_lamp("#ffd34a", "#fffbe0"),
        "green": bulb_lamp("#3ee06a", "#e2ffe9"),
        "red": bulb_lamp("#ff3b3b", "#ffe2e2"),
        "glow": "255, 214, 90",
    },
    "newyear": {
        "off": star_lamp("#4d401a", "#5e4f24"),
        "on": star_lamp("#ffd75a", "#fffbe2"),
        "green": star_lamp("#3ee6e0", "#e8fffe"),
        "red": star_lamp("#ff4fb4", "#ffe8f5"),
        "glow": "255, 222, 110",
    },
}

# ------------------------------------------------------------------ stickers (64 × 64, a white sticker border)
W = "stroke='#fff' stroke-width='3.2' stroke-linejoin='round' paint-order='stroke'"
STICKERS = {
    "halloween": [
        # a jack-o'-lantern, grinning
        S(f"""<path d='M32 12c1-5 4-8 8-9' stroke='#3d6b1f' stroke-width='4' fill='none' stroke-linecap='round'/>
<g {W}><ellipse cx='20' cy='37' rx='15' ry='20' fill='#ff7a12'/><ellipse cx='44' cy='37' rx='15' ry='20' fill='#ff7a12'/>
<ellipse cx='32' cy='37' rx='14' ry='21' fill='#ff9028'/></g>
<path d='M15 30l7-8 5 9zM49 30l-7-8-5 9z' fill='#ffe86a'/>
<path d='M14 42h36l-4 8-4-3-4 4-4-4-4 4-4-4-4 3z' fill='#ffe86a'/>
<path d='M15 30l7-8 5 9zM49 30l-7-8-5 9zM14 42h36l-4 8-4-3-4 4-4-4-4 4-4-4-4 3z' fill='none' stroke='#7a2c00' stroke-width='1.4'/>"""),
        # a bat
        S(f"""<path {W} fill='#2a1238' d='M32 24c2-4 3-6 3-9 2 2 3 4 3 7 6-6 14-7 22-4-5 3-6 8-5 13-4-3-8-3-11 0 0-3-2-5-5-5-1 4-4 8-7 10-3-2-6-6-7-10-3 0-5 2-5 5-3-3-7-3-11 0 1-5 0-10-5-13 8-3 16-2 22 4 0-3 1-5 3-7 0 3 1 5 3 9z'/>
<circle cx='29.5' cy='27' r='1.6' fill='#ffd93a'/><circle cx='34.5' cy='27' r='1.6' fill='#ffd93a'/>"""),
        # a little ghost, waving
        S(f"""<path {W} fill='#f6f3ff' d='M16 54V30a16 16 0 0 1 32 0v24l-5-4-5 4-6-4-6 4-5-4z'/>
<ellipse cx='26' cy='30' rx='2.6' ry='3.6' fill='#2a1238'/><ellipse cx='38' cy='30' rx='2.6' ry='3.6' fill='#2a1238'/>
<ellipse cx='32' cy='39' rx='3' ry='2.4' fill='#2a1238'/><circle cx='22' cy='35' r='2' fill='#ffb3c6' opacity='.7'/><circle cx='42' cy='35' r='2' fill='#ffb3c6' opacity='.7'/>"""),
        # a witch's hat
        S(f"""<g {W}><path fill='#3a1a52' d='M10 48c8-4 36-4 44 0-6 4-38 4-44 0z'/><path fill='#4c2470' d='M20 47c2-10 6-26 18-38-2 10 2 18 6 22l-4 16z'/></g>
<path d='M20.5 43c6 1.5 14 1.5 19.8 0l-.8 4c-6 1-12 1-18.4 0z' fill='#ff8a1a'/><rect x='27' y='41.5' width='6' height='5.5' rx='1' fill='#ffd93a'/>"""),
        # candy corn
        S(f"""<g {W}><path fill='#fff6e0' d='M32 8c8 10 14 26 14 40-6 3-22 3-28 0 0-14 6-30 14-40z'/></g>
<path fill='#ff8a1a' d='M21 28c6-1.5 16-1.5 22 0 1.8 6 2.8 13 3 20-6 3-22 3-28 0 .2-7 1.2-14 3-20z'/>
<path fill='#ffd23a' d='M19 40c8-1.5 18-1.5 26 0 .6 2.6 1 5.4 1 8-6 3-22 3-28 0 0-2.6.4-5.4 1-8z'/>"""),
    ],
    "thanksgiving": [
        # a round little turkey with a fan tail
        S(f"""<g {W}><path fill='#c0392b' d='M32 34L10 22a24 24 0 0 1 6-10z'/><path fill='#e67e22' d='M32 34L16 12a24 24 0 0 1 9-5z'/>
<path fill='#f1c40f' d='M32 34L25 7a24 24 0 0 1 14 0z'/><path fill='#e67e22' d='M32 34L39 7a24 24 0 0 1 9 5z'/><path fill='#c0392b' d='M32 34L48 12a24 24 0 0 1 6 10z'/>
<ellipse cx='32' cy='42' rx='14' ry='14' fill='#7a4a24'/><circle cx='32' cy='26' r='8' fill='#8f5a2c'/></g>
<circle cx='29' cy='25' r='1.5' fill='#1b1208'/><circle cx='35' cy='25' r='1.5' fill='#1b1208'/><path d='M30 28l2 3 2-3z' fill='#f5b301'/>
<path d='M33 30c1 2 1 4-.5 5' stroke='#d6322a' stroke-width='2' fill='none' stroke-linecap='round'/>
<path d='M26 55l-2 4M38 55l2 4' stroke='#f5b301' stroke-width='2.4' stroke-linecap='round'/>"""),
        # a maple leaf
        S(f"""<path {W} fill='#e8582a' d='M32 6l4 9 7-3-2 10 9-2-4 8 7 4-11 4 2 6-10-2-1 10h-2l-1-10-10 2 2-6-11-4 7-4-4-8 9 2-2-10 7 3z'/>
<path d='M32 18v38' stroke='#8a2a0e' stroke-width='1.8'/>"""),
        # a slice of pumpkin pie with cream
        S(f"""<g {W}><path fill='#d9822b' d='M8 44l46-20 2 14-48 18z'/><path fill='#c9894a' d='M8 44l48-6v6L8 56z'/></g>
<path fill='#e9a24a' d='M10 43.5l44-19.3 1 5.6-43 14.7z'/><path fill='#fffaf0' stroke='#e8dcc8' stroke-width='1' d='M38 25c-2-6 6-9 9-5 3-3 9 0 6 5-4 3-11 3-15 0z'/>"""),
        # an acorn
        S(f"""<g {W}><path fill='#b8743a' d='M18 30h28c0 14-6 26-14 28-8-2-14-14-14-28z'/><path fill='#6b4422' d='M14 30c0-10 8-14 18-14s18 4 18 14z'/></g>
<path d='M32 16c0-5 2-8 6-10' stroke='#6b4422' stroke-width='3' fill='none' stroke-linecap='round'/>
<path d='M18 24h28M20 28h24' stroke='#4f3016' stroke-width='1.2'/><ellipse cx='26' cy='40' rx='2.5' ry='6' fill='#e3a76a' opacity='.6'/>"""),
        # a cob of corn
        S(f"""<g {W}><path fill='#7aa83a' d='M18 56c-2-16 0-34 10-44-2 14 0 30 6 44z'/><path fill='#f4c430' d='M24 54c-4-14 0-36 12-46 8 6 8 30-2 46z'/>
<path fill='#8fbf45' d='M46 56c2-16 0-34-10-44 2 14 0 30-6 44z'/></g>
<g fill='#e0a91c'><circle cx='31' cy='22' r='1.6'/><circle cx='35' cy='24' r='1.6'/><circle cx='30' cy='28' r='1.6'/><circle cx='34' cy='30' r='1.6'/><circle cx='30' cy='35' r='1.6'/><circle cx='34' cy='37' r='1.6'/><circle cx='30' cy='42' r='1.6'/><circle cx='33' cy='45' r='1.6'/></g>"""),
    ],
    "christmas": [
        # a decorated tree with a star
        S(f"""<g {W}><path fill='#1f8a4c' d='M32 8L50 34h-7l9 14H12l9-14h-7z'/><rect x='28' y='48' width='8' height='8' fill='#7a4a24'/></g>
<path d='M32 3l1.8 3.8 4.2.4-3.2 2.8 1 4.1L32 12l-3.8 2.1 1-4.1-3.2-2.8 4.2-.4z' fill='#ffd84a' stroke='#fff' stroke-width='1.2'/>
<circle cx='28' cy='26' r='2.2' fill='#e8323a'/><circle cx='37' cy='31' r='2.2' fill='#ffd84a'/><circle cx='24' cy='40' r='2.4' fill='#3aa6ff'/>
<circle cx='34' cy='42' r='2.4' fill='#e8323a'/><circle cx='42' cy='44' r='2.2' fill='#ffd84a'/>
<path d='M22 34c6 3 14 3 20-2M18 46c8 3 20 3 28-2' stroke='#fff7c2' stroke-width='1.4' fill='none'/>"""),
        # a candy cane
        S(f"""<path d='M40 58V22a10 10 0 0 0-20 0' stroke='#fff' stroke-width='14' fill='none' stroke-linecap='round'/>
<path d='M40 58V22a10 10 0 0 0-20 0' stroke='#fbfbfb' stroke-width='9' fill='none' stroke-linecap='round'/>
<path d='M40 58V22a10 10 0 0 0-20 0' stroke='#e02b35' stroke-width='9' fill='none' stroke-dasharray='5 5' stroke-linecap='butt'/>"""),
        # a bauble
        S(f"""<g {W}><circle cx='32' cy='38' r='18' fill='#d62839'/><rect x='27' y='15' width='10' height='6' rx='1.5' fill='#d4af37'/></g>
<path d='M32 15c0-4 3-6 5-6' stroke='#c9a227' stroke-width='1.8' fill='none'/>
<path d='M15 34c10 5 24 5 34 0' stroke='#ffd84a' stroke-width='3' fill='none'/><path d='M16 42c10 4 22 4 32 0' stroke='#fff' stroke-width='1.6' fill='none' stroke-dasharray='2 3'/>
<ellipse cx='25' cy='30' rx='3' ry='5' fill='#fff' opacity='.45' transform='rotate(-25 25 30)'/>"""),
        # holly
        S(f"""<g {W}><path fill='#1f8a4c' d='M30 34c-6-8-16-8-24-4 4 2 2 6 6 7-2 3 1 6 4 6 0 4 4 4 6 2 2 2 6 0 8-11z'/>
<path fill='#26a05a' d='M34 34c6-8 16-8 24-4-4 2-2 6-6 7 2 3-1 6-4 6 0 4-4 4-6 2-2 2-6 0-8-11z'/></g>
<circle cx='28' cy='36' r='5' fill='#e02b35' stroke='#fff' stroke-width='2'/><circle cx='36' cy='36' r='5' fill='#e02b35' stroke='#fff' stroke-width='2'/><circle cx='32' cy='29' r='5' fill='#e8323a' stroke='#fff' stroke-width='2'/>"""),
        # a present
        S(f"""<g {W}><rect x='12' y='28' width='40' height='26' rx='2' fill='#2b7de0'/><rect x='10' y='22' width='44' height='9' rx='2' fill='#3a8cf0'/></g>
<rect x='29' y='22' width='6' height='32' fill='#ffd84a'/><path d='M32 22c-6-8-14-8-12-2 2 4 8 2 12 2 4 0 10 2 12-2 2-6-6-6-12 2z' fill='#ffd84a' stroke='#c9a227' stroke-width='1'/>"""),
        # a snowflake
        S(f"""<g stroke='#fff' stroke-width='7' stroke-linecap='round'><path d='M32 6v52M9.5 19l45 26M9.5 45l45-26'/></g>
<g stroke='#7cc4ff' stroke-width='3.4' stroke-linecap='round' fill='none'><path d='M32 6v52M9.5 19l45 26M9.5 45l45-26'/>
<path d='M26 10l6 6 6-6M26 54l6-6 6 6M12 26l8-2-2-8M52 38l-8 2 2 8M12 38l8 2-2 8M52 26l-8-2 2-8'/></g>"""),
    ],
    "newyear": [
        # two glasses clinking
        S(f"""<g {W}><path fill='#f7e7a6' d='M10 12h16l-2 16a6 6 0 0 1-12 0z'/><path fill='#f7e7a6' d='M38 12h16l-2 16a6 6 0 0 1-12 0z'/></g>
<path d='M12 18h12l-1.2 9a4.6 4.6 0 0 1-9.4 0zM40 18h12l-1.2 9a4.6 4.6 0 0 1-9.4 0z' fill='#ffd24a'/>
<path d='M18 34v16M46 34v16M12 52h12M40 52h12' stroke='#fff' stroke-width='4.5' stroke-linecap='round'/>
<path d='M18 34v16M46 34v16M12 52h12M40 52h12' stroke='#d8c27a' stroke-width='2.2' stroke-linecap='round'/>
<path d='M32 4v6M26 6l3 5M38 6l-3 5' stroke='#ff4fb4' stroke-width='2.4' stroke-linecap='round'/>
<circle cx='17' cy='22' r='1.2' fill='#fff'/><circle cx='47' cy='24' r='1.2' fill='#fff'/><circle cx='20' cy='26' r='.9' fill='#fff'/>"""),
        # a firework burst
        S(f"""<g stroke-linecap='round' stroke-width='3.4'><g stroke='#ff4fb4'><path d='M32 32L32 6M32 32L50 14M32 32L58 32M32 32L50 50M32 32L32 58M32 32L14 50M32 32L6 32M32 32L14 14'/></g>
<g stroke='#3ee6e0' stroke-width='2.2'><path d='M32 32l9-21M32 32l21 9M32 32l-9 21M32 32l-21-9M32 32l21-9M32 32l-9-21M32 32l-21 9M32 32l9 21'/></g></g>
<circle cx='32' cy='32' r='5' fill='#ffd75a' stroke='#fff' stroke-width='2'/><g fill='#ffd75a'><circle cx='32' cy='4' r='2.4'/><circle cx='60' cy='32' r='2.4'/><circle cx='32' cy='60' r='2.4'/><circle cx='4' cy='32' r='2.4'/></g>"""),
        # a party hat
        S(f"""<g {W}><path fill='#7b4dff' d='M32 6L50 54H14z'/></g>
<path d='M24 28l16 2M20 40l24 2M17 50l30 1' stroke='#ffd75a' stroke-width='3'/><circle cx='26' cy='34' r='2' fill='#3ee6e0'/><circle cx='38' cy='46' r='2' fill='#ff4fb4'/>
<circle cx='32' cy='6' r='5' fill='#ffd75a' stroke='#fff' stroke-width='2'/><path d='M12 56c6-3 34-3 40 0' stroke='#ff4fb4' stroke-width='4' fill='none' stroke-linecap='round'/>"""),
        # a clock at midnight
        S(f"""<g {W}><circle cx='32' cy='34' r='22' fill='#1c1f3f'/></g><circle cx='32' cy='34' r='18' fill='#f7f1dc'/>
<g stroke='#1c1f3f' stroke-width='2' stroke-linecap='round'><path d='M32 18v4M32 46v4M16 34h4M44 34h4'/></g>
<path d='M32 34V22M32 34l-1-13' stroke='#1c1f3f' stroke-width='3' stroke-linecap='round'/><circle cx='32' cy='34' r='2.4' fill='#ff4fb4'/>
<path d='M14 14l6 4M50 14l-6 4' stroke='#ffd75a' stroke-width='3' stroke-linecap='round'/>"""),
        # streamers and confetti
        S(f"""<path d='M8 12c8 6 0 12 8 18s0 12 8 18' stroke='#fff' stroke-width='7' fill='none' stroke-linecap='round'/>
<path d='M8 12c8 6 0 12 8 18s0 12 8 18' stroke='#ff4fb4' stroke-width='4' fill='none' stroke-linecap='round'/>
<path d='M40 8c8 6 0 12 8 18s0 12 8 18' stroke='#fff' stroke-width='7' fill='none' stroke-linecap='round'/>
<path d='M40 8c8 6 0 12 8 18s0 12 8 18' stroke='#3ee6e0' stroke-width='4' fill='none' stroke-linecap='round'/>
<g stroke='#fff' stroke-width='1.2'><rect x='28' y='14' width='5' height='5' fill='#ffd75a' transform='rotate(20 30 16)'/><rect x='30' y='38' width='5' height='5' fill='#7b4dff' transform='rotate(-15 32 40)'/>
<circle cx='26' cy='28' r='2.8' fill='#3ee6e0'/><circle cx='38' cy='52' r='2.8' fill='#ff4fb4'/><circle cx='52' cy='54' r='2.4' fill='#ffd75a'/></g>"""),
    ],
}

# ------------------------------------------------------------------ the meter (160 × 90): off and lit pictures
def jack_meter(lit):
    face = "url(#fire)" if lit else "#1a0a02"
    halo = ("<radialGradient id='halo' cx='.5' cy='.58' r='.5'><stop offset='0' stop-color='#ffb84a' stop-opacity='.75'/>"
            "<stop offset='1' stop-color='#ff8a1a' stop-opacity='0'/></radialGradient>") if lit else ""
    return S(f"""<defs><linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#160a26'/><stop offset='1' stop-color='#3a1a52'/></linearGradient>
<radialGradient id='fire' cx='.5' cy='.6' r='.6'><stop offset='0' stop-color='#fffbd0'/><stop offset='.45' stop-color='#ffd84a'/><stop offset='1' stop-color='#ff8a1a'/></radialGradient>
<radialGradient id='skin' cx='.45' cy='.4' r='.7'><stop offset='0' stop-color='{"#ffa13a" if lit else "#c8641c"}'/><stop offset='1' stop-color='{"#d8620c" if lit else "#7a3a0e"}'/></radialGradient>{halo}</defs>
<rect width='160' height='90' fill='url(#sky)'/>
<circle cx='130' cy='18' r='10' fill='#f4ecc8' opacity='{".95" if lit else ".55"}'/><circle cx='134' cy='15' r='9' fill='#1d0e30' opacity='.9'/>
<g fill='#f4ecc8' opacity='.7'><circle cx='20' cy='14' r='.9'/><circle cx='44' cy='8' r='.7'/><circle cx='106' cy='10' r='.8'/><circle cx='150' cy='40' r='.7'/><circle cx='10' cy='44' r='.7'/></g>
<path d='M14 30l3-2 3 2 1-3 2 3 3-2' stroke='#0b0612' stroke-width='1.6' fill='none'/>
{"<ellipse cx='80' cy='58' rx='62' ry='34' fill='url(#halo)'/>" if lit else ""}
<path d='M80 24c1-6 4-9 9-11' stroke='#3d6b1f' stroke-width='4.5' fill='none' stroke-linecap='round'/>
<ellipse cx='62' cy='58' rx='24' ry='27' fill='url(#skin)'/><ellipse cx='98' cy='58' rx='24' ry='27' fill='url(#skin)'/><ellipse cx='80' cy='58' rx='21' ry='29' fill='url(#skin)'/>
<path d='M68 34c-3 8-3 40 0 48M92 34c3 8 3 40 0 48' stroke='#000' stroke-opacity='.18' stroke-width='1.6' fill='none'/>
<path d='M57 50l9-11 7 12zM103 50l-9-11-7 12zM76 56l4-6 4 6z' fill='{face}'/>
<path d='M52 64h56l-6 12-6-5-5 6-5-6-5 6-5-6-5 6-6-5z' fill='{face}'/>
<rect x='0' y='84' width='160' height='6' fill='#0b0612'/>""", "0 0 160 90")


def globe_meter(lit):
    win = "#ffd34a" if lit else "#24304a"
    return S(f"""<defs><radialGradient id='glass' cx='.4' cy='.35' r='.7'><stop offset='0' stop-color='{"#3a5a9a" if lit else "#1d2a4a"}'/><stop offset='1' stop-color='{"#16264e" if lit else "#0c1428"}'/></radialGradient>
<radialGradient id='warm' cx='.5' cy='.5' r='.5'><stop offset='0' stop-color='#ffcf5a' stop-opacity='.7'/><stop offset='1' stop-color='#ffcf5a' stop-opacity='0'/></radialGradient></defs>
<rect width='160' height='90' fill='#0e1a12'/>
<circle cx='80' cy='42' r='36' fill='url(#glass)'/>
<g fill='#fff' opacity='.85'><circle cx='60' cy='22' r='1.3'/><circle cx='92' cy='18' r='1'/><circle cx='104' cy='34' r='1.2'/><circle cx='54' cy='40' r='1'/><circle cx='72' cy='14' r='.9'/></g>
<path d='M46 60c10-6 58-6 68 0v6H46z' fill='#eef4ff'/>
<path d='M98 58l8-22 8 22z' fill='#1f7a44'/><path d='M100 50l6-14 6 14z' fill='#26944f'/><path d='M101 40l5-9 5 9z' fill='#eef4ff'/>
{"<ellipse cx='72' cy='50' rx='22' ry='16' fill='url(#warm)'/>" if lit else ""}
<rect x='58' y='42' width='28' height='18' fill='#8a4a24'/><path d='M54 44l18-14 18 14z' fill='#eef4ff'/><path d='M56 44l16-12 16 12' stroke='#6a3418' stroke-width='1.4' fill='none'/>
<rect x='78' y='31' width='5' height='8' fill='#6a3418'/><rect x='77' y='29' width='7' height='3' fill='#eef4ff'/>
<rect x='62' y='46' width='7' height='6' fill='{win}'/><rect x='75' y='46' width='7' height='6' fill='{win}'/><path d='M65.5 46v6M62 49h7M78.5 46v6M75 49h7' stroke='#6a3418' stroke-width='.8'/>
<rect x='70' y='50' width='5' height='10' fill='#5a2e14'/>
{"<path d='M80 26c-3-4 3-6 0-10s3-6 0-9' stroke='#fff' stroke-opacity='.5' stroke-width='2' fill='none' stroke-linecap='round'/>" if lit else ""}
<path d='M58 22a30 30 0 0 1 28-9' stroke='#fff' stroke-opacity='.35' stroke-width='3' fill='none' stroke-linecap='round'/>
<circle cx='80' cy='42' r='36' fill='none' stroke='#dbe8ff' stroke-opacity='.6' stroke-width='2'/>
<path d='M44 74c4-6 68-6 72 0l6 14H38z' fill='#b3242c'/><path d='M44 74c4-6 68-6 72 0' stroke='#ffd84a' stroke-width='2' fill='none'/>
<path d='M50 82h60' stroke='#ffd84a' stroke-width='1.4' stroke-dasharray='3 3'/>""", "0 0 160 90")


METERS = {"halloween": (jack_meter(False), jack_meter(True), "255, 170, 40"),
          "christmas": (globe_meter(False), globe_meter(True), "255, 210, 90")}

# ------------------------------------------------------------------ colours (Light faceplate / Dark faceplate)
PALETTE = {
    "halloween": dict(
        lit=("#fff7c8", "#ffb03a", "#f07a12", "#2a1200", "#5a2800", "#6e3200"), glow="255, 150, 40", accent="#ff8a1a", accent_rgb="255, 138, 26", accent2="#9b5cff", amber="#ffb03a",
        key=("#8e5ad8", "#5e2fa8", "#40197a", "#1e0b3a"), keylit=("#fff7c8", "#ffb03a", "#f07a12", "#2a1200"),
        light=dict(card=("#f0a25a", "#e48a3c", "#ea9446"), ear=("#3a1a52", "#6e3aa0", "#55288a", "#2c1240", "#170820"), stripe=("#9b5cff", "#6e3aa0", "#3a1a52"), orange="#4c1d70", orange_deep="#2a0e42"),
        dark=dict(card=("#2e1d3c", "#22152e", "#281933"), panel="#2a1a36"),
        swatch="linear-gradient(90deg, #2a1238 0 40%, #ff8a1a 40% 75%, #8cff5a 75%)"),
    "thanksgiving": dict(
        lit=("#fff4d6", "#ffc04a", "#e08a1e", "#2a1600", "#5a3200", "#6e3e00"), glow="255, 176, 50", accent="#e8742a", accent_rgb="232, 116, 42", accent2="#9c2a3a", amber="#ffc04a",
        key=("#c47a46", "#9a5226", "#723a18", "#341808"), keylit=("#fff4d6", "#ffc04a", "#e08a1e", "#2a1600"),
        light=dict(card=("#e6c896", "#d8b47a", "#dfbe88"), ear=("#6a2a1c", "#a8442c", "#8a3624", "#541f14", "#2a0e08"), stripe=("#e8742a", "#b8501c", "#7a3010"), orange="#8a2f14", orange_deep="#5c1e0a"),
        dark=dict(card=("#3e2818", "#301e12", "#372315"), panel="#3a2416"),
        swatch="linear-gradient(90deg, #5a341a 0 40%, #e8742a 40% 72%, #ffc04a 72% 88%, #9c2a3a 88%)"),
    "christmas": dict(
        lit=("#fffbe6", "#ffd84a", "#e0a81a", "#2a2000", "#5a4600", "#6e5600"), glow="255, 214, 90", accent="#e02b35", accent_rgb="224, 43, 53", accent2="#1f8a4c", amber="#ffd84a",
        key=("#e8565c", "#c42630", "#951a22", "#46080c"), keylit=("#fffbe6", "#ffd84a", "#e0a81a", "#2a2000"),
        light=dict(card=("#f6f7fa", "#e6e9ef", "#eef0f4"), ear=("#145c34", "#2a9a5a", "#1f7f48", "#0e4426", "#062010"), stripe=("#e02b35", "#b51e28", "#7e121a"), orange="#b51e28", orange_deep="#145c34"),
        dark=dict(card=("#16402a", "#0f3020", "#133825"), panel="#123622"),
        swatch="linear-gradient(90deg, #145c34 0 38%, #f4f6fa 38% 60%, #e02b35 60% 82%, #ffd84a 82%)"),
    "newyear": dict(
        lit=("#fffbe2", "#ffd75a", "#d9a81e", "#2a2000", "#5a4600", "#6e5600"), glow="255, 222, 110", accent="#ff4fb4", accent_rgb="255, 79, 180", accent2="#3ee6e0", amber="#ffd75a",
        key=("#5a5ec8", "#33379a", "#22256e", "#0c0e30"), keylit=("#fffbe2", "#ffd75a", "#d9a81e", "#2a2000"),
        light=dict(card=("#ece0bc", "#ddcd9e", "#e4d6ac"), ear=("#1c1f3f", "#3a3f7a", "#2c3062", "#141630", "#080a1a"), stripe=("#ff4fb4", "#d6339a", "#7b4dff"), orange="#5a2a9a", orange_deep="#1c1f3f"),
        dark=dict(card=("#1d2048", "#151838", "#191c40"), panel="#181b3e"),
        swatch="linear-gradient(90deg, #121430 0 40%, #ffd75a 40% 70%, #ff4fb4 70% 85%, #3ee6e0 85%)"),
}

# where the stickers sit on the top panel (left, top in % of the panel, size px, turn): over blank areas only
# where the stickers sit on the top panel: on parts that never hold words or controls (the two rack ears, the logo
# plate's corner and the VU meter's corner), placed from the panel's edges so window size doesn't move them onto a label
SPOTS = [("left: -4px; top: 2px", 46, -14), ("left: 286px; top: 50px", 30, 10), ("right: 44px; top: -4px", 36, -8),
         ("right: -2px; top: 54px", 34, 12), ("left: 2px; top: 96px", 28, -6)]
CORNER = (54, -10)  # the big one in the window's bottom right corner


def main():
    css = ["/* Holiday looks: made by packaging/skins/build_holiday_css.py (edit the drawings there, then run it). All art is",
           "   original, drawn for Protoart Studio. */", ""]
    for k, P in PALETTE.items():
        r = f':root[data-skin="{k}"]'
        lit, key, keylit = P["lit"], P["key"], P["keylit"]
        css.append(f"""{r} {{
  --lit-hi: {lit[0]}; --lit-mid: {lit[1]}; --lit-lo: {lit[2]}; --lit-ink: {lit[3]}; --lit-ink2: {lit[4]}; --lit-edge: {lit[5]};
  --glow-rgb: {P['glow']}; --accent-rgb: {P['accent_rgb']}; --accent: {P['accent']}; --accent-2: {P['accent2']}; --amber: {P['amber']};
  --key-hi: {key[0]}; --key-mid: {key[1]}; --key-lo: {key[2]}; --key-edge: {key[3]};
  --key-lit-hi: {keylit[0]}; --key-lit-mid: {keylit[1]}; --key-lit-lo: {keylit[2]}; --key-lit-ink: {keylit[3]};
}}""")
        L, D = P["light"], P["dark"]
        css.append(f"""{r}[data-theme="light"] {{
  --plate-card: linear-gradient(180deg, {L['card'][0]}, {L['card'][1]}) 0 0 / 100% 100%, {L['card'][2]};
  --orange: {L['orange']}; --orange-deep: {L['orange_deep']};
  --ear-1: {L['ear'][0]}; --ear-2: {L['ear'][1]}; --ear-3: {L['ear'][2]}; --ear-4: {L['ear'][3]}; --ear-edge: {L['ear'][4]};
  --stripe-1: {L['stripe'][0]}; --stripe-2: {L['stripe'][1]}; --stripe-3: {L['stripe'][2]};
}}
{r}[data-theme="dark"] {{
  --plate-card: linear-gradient(180deg, {D['card'][0]}, {D['card'][1]}) 0 0 / 100% 100%, {D['card'][2]};
  --panel: {D['panel']};
}}
.skin-swatch.{k} {{ background: {P['swatch']}; }}""")
        # lamps: the shapes replace the round LEDs everywhere (tabs, Power / Busy / Fault, the launch screen)
        Lm = LAMPS[k]
        css.append(f"""{r} .led, {r} #splash .sp-led {{ width: 13px; height: 13px; border-radius: 0; box-shadow: none;
  background: {uri(Lm['off'])} center / contain no-repeat; }}
{r} .led.on, {r} nav button.active .led, {r} .sw-pos.on .sw-led {{ background: {uri(Lm['on'])} center / contain no-repeat;
  box-shadow: none; filter: drop-shadow(0 0 3px rgba({Lm['glow']}, .95)) drop-shadow(0 0 7px rgba({Lm['glow']}, .55)); }}
{r} .led.green.on {{ background-image: {uri(Lm['green'])}; }}
{r} .led.red.on {{ background-image: {uri(Lm['red'])}; }}
{r} .led.blink {{ animation: hdBlink 1.1s steps(1) infinite; }}
{r} #splash .sp-power, {r} #splash.ready .sp-ready {{ animation: none; background-image: {uri(Lm['green'])}; filter: drop-shadow(0 0 4px rgba({Lm['glow']}, .9)); }}
{r} #splash .sp-engine {{ animation: hdEngine-{k} .9s .7s infinite alternate both; }}
{r} #splash.ready .sp-engine {{ animation: none; background-image: {uri(Lm['on'])}; filter: drop-shadow(0 0 4px rgba({Lm['glow']}, .9)); }}
@keyframes hdEngine-{k} {{ from {{ background-image: {uri(Lm['off'])}; filter: none; }} to {{ background-image: {uri(Lm['on'])}; filter: drop-shadow(0 0 5px rgba({Lm['glow']}, .95)); }} }}""")
        # stickers
        st = STICKERS[k]
        for i, (where, size, rot) in enumerate(SPOTS):
            css.append(f"{r} .holiday-deco .hd{i} {{ display: block; {where}; width: {size}px; height: {size}px; transform: rotate({rot}deg); background-image: {uri(st[i % len(st)])}; }}")
        css.append(f"{r} .holiday-corner {{ display: block; width: {CORNER[0]}px; height: {CORNER[0]}px; transform: rotate({CORNER[1]}deg); background-image: {uri(st[0])}; }}")
        css.append(f"{r} #splash .sp-face::after {{ content: ''; width: 64px; height: 64px; right: -14px; top: -22px; border-radius: 0; box-shadow: none; transform: rotate(12deg);"
                   f" background: {uri(st[0])} center / contain no-repeat; filter: drop-shadow(0 2px 3px rgba(0, 0, 0, .45)); }}")
        if k in METERS:  # the VU meter becomes a picture that lights up
            off, on, glow = METERS[k]
            css.append(f"""{r} .rack-vu .vu-holiday {{ display: block; }}
{r} .rack-vu svg, {r} .rack-vu .vu-lamp {{ visibility: hidden; }}
{r} .rack-vu .vh-off {{ background-image: {uri(off)}; }}
{r} .rack-vu .vh-on {{ background-image: {uri(on)}; }}
{r} .rack-vu.lit .vh-on {{ opacity: .12; }}
{r} .rack-vu.live .vh-on {{ opacity: calc(.45 + var(--vu, 0) * .55); animation: vhGlow-{k} 1.6s ease-in-out infinite; }}
{r} .rack-vu .vh-bar {{ background: rgba({glow}, .95); box-shadow: 0 0 6px rgba({glow}, .9); }}
@keyframes vhGlow-{k} {{ 50% {{ filter: brightness(1.12) saturate(1.1); }} }}""")
        css.append("")
    css.append("""/* the holiday meters: off picture, lit picture over it, snow in the globe, the progress line */
.vu-holiday { display: none; position: absolute; inset: 0; pointer-events: none; }
.vu-holiday i { position: absolute; inset: 0; background: center / 100% 100% no-repeat; }
.vu-holiday .vh-on { opacity: 0; transition: opacity 1.2s ease; }
.vu-holiday .vh-snow, .vu-holiday .vh-bar { display: none; }
.rack-vu.live .vu-holiday .vh-bar { display: block; top: auto; bottom: 0; height: 3px; right: auto; transition: width .6s ease; }
:root[data-skin="christmas"] .rack-vu .vh-snow { display: block; left: 27.5%; right: 27.5%; top: 6%; bottom: 14%; border-radius: 50%; opacity: 0;
  background-image: radial-gradient(1.6px 1.6px at 10px 8px, #fff 50%, transparent 52%), radial-gradient(1.2px 1.2px at 30px 22px, #fff 50%, transparent 52%),
    radial-gradient(1.8px 1.8px at 50px 12px, #fff 50%, transparent 52%), radial-gradient(1.2px 1.2px at 22px 36px, #fff 50%, transparent 52%),
    radial-gradient(1.5px 1.5px at 60px 40px, #fff 50%, transparent 52%);
  background-size: 70px 50px; background-repeat: repeat; transition: opacity 1s; }
:root[data-skin="christmas"] .rack-vu.live .vh-snow { opacity: .95; animation: vhSnow 3.5s linear infinite; }
@keyframes vhSnow { from { background-position: 0 -50px; } to { background-position: 14px 0; } }
/* Halloween: the lit jack-o'-lantern flickers like a candle inside it */
:root[data-skin="halloween"] .rack-vu.live .vh-on { animation: vhCandle 2.2s infinite; }
@keyframes vhCandle { 0%, 100% { filter: brightness(1); } 30% { filter: brightness(.82); } 33% { filter: brightness(1.15); } 60% { filter: brightness(.93); } }
@media (prefers-reduced-motion: reduce) { .vu-holiday * { animation: none !important; } }

/* the stickers: only in the holiday looks, never in the way of a click */
.holiday-deco { position: absolute; inset: 0; pointer-events: none; z-index: 3; }
.holiday-deco i, .holiday-corner { display: none; position: absolute; background: center / contain no-repeat; pointer-events: none;
  filter: drop-shadow(0 2px 2px rgba(0, 0, 0, .45)); }
.holiday-corner { position: fixed; right: 14px; bottom: 10px; z-index: 40; opacity: .95; }
.rack-head { position: relative; }
@keyframes hdBlink { 50% { filter: brightness(.45); } }
/* Halloween: the jack-o'-lantern on the panel flickers like a candle inside it */
:root[data-skin="halloween"] .holiday-deco .hd0 { animation: hdFlicker 3.2s infinite; }
@keyframes hdFlicker { 0%, 100% { filter: drop-shadow(0 0 6px rgba(255, 170, 40, .9)) drop-shadow(0 2px 2px rgba(0, 0, 0, .45)); }
  40% { filter: drop-shadow(0 0 2px rgba(255, 150, 30, .5)) drop-shadow(0 2px 2px rgba(0, 0, 0, .45)); }
  44% { filter: drop-shadow(0 0 9px rgba(255, 190, 60, 1)) drop-shadow(0 2px 2px rgba(0, 0, 0, .45)); } }
/* Christmas: soft snow over the top panel; New Year's: the stars and firework twinkle */
:root[data-skin="christmas"] .holiday-deco::before { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .75;
  background-image: radial-gradient(2px 2px at 20px 30px, #fff 50%, transparent 51%), radial-gradient(2.5px 2.5px at 90px 80px, #fff 50%, transparent 51%),
    radial-gradient(1.6px 1.6px at 150px 20px, #fff 50%, transparent 51%), radial-gradient(2px 2px at 210px 110px, #fff 50%, transparent 51%),
    radial-gradient(1.8px 1.8px at 260px 60px, #fff 50%, transparent 51%);
  background-size: 300px 160px; animation: hdSnow 9s linear infinite; mix-blend-mode: screen; }
@keyframes hdSnow { from { background-position: 0 -160px; } to { background-position: 40px 0; } }
:root[data-skin="newyear"] .holiday-deco .hd1, :root[data-skin="newyear"] .holiday-corner { animation: hdTwinkle 2.4s ease-in-out infinite; }
@keyframes hdTwinkle { 50% { transform: rotate(-6deg) scale(1.08); filter: drop-shadow(0 0 8px rgba(255, 222, 110, .9)) drop-shadow(0 2px 2px rgba(0, 0, 0, .45)); } }
/* Thanksgiving: a leaf drifts in the corner */
:root[data-skin="thanksgiving"] .holiday-deco .hd1 { animation: hdDrift 6s ease-in-out infinite; }
@keyframes hdDrift { 50% { transform: rotate(24deg) translateY(4px); } }
@media (prefers-reduced-motion: reduce) { .holiday-deco *, .holiday-deco::before, .holiday-corner { animation: none !important; } }
/* the holiday look buttons in Settings */
.holiday-pick button { padding: 4px 8px; font-size: 15px; line-height: 1; }
""")
    OUT.write_text("\n".join(css))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
