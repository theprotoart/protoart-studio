"""Procreate files: LZO tiles, the document, the flattened picture and the layers (a tiny made-up file)."""
import io
import os
import plistlib
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import procreate  # noqa: E402


def lzo_literals(data: bytes) -> bytes:
    """the simplest valid LZO1X stream: one literal run, then the end marker"""
    assert len(data) <= 238
    return bytes([17 + len(data)]) + data + b"\x11\x00\x00"


def make_file(path, w=6, h=5, tile=4):
    U = plistlib.UID
    layer = lambda uuid, name, op, hidden: {"UUID": U(5 + 0) if False else uuid, "name": name, "opacity": op, "hidden": hidden,
                                             "blend": 0, "clipped": False}
    objs = ["$null",
            {"size": U(2), "layers": U(3), "composite": U(6), "tileSize": tile, "orientation": 1, "name": U(7),
             "flippedHorizontally": False, "flippedVertically": False, "backgroundHidden": False,
             "backgroundColor": U(8)},
            "{%d, %d}" % (w, h),
            {"NS.objects": [U(4), U(5)]},
            {"UUID": "TOP", "name": "Ink", "opacity": 1.0, "hidden": False, "blend": 0, "clipped": False},
            {"UUID": "BOT", "name": "Paint", "opacity": 0.5, "hidden": False, "blend": 1, "clipped": False},
            {"UUID": "COMP"},
            "Tiny",
            np.array([1, 1, 1, 1], "<f4").tobytes()]
    arch = {"$archiver": "NSKeyedArchiver", "$version": 100000, "$top": {"root": U(1)}, "$objects": objs}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Document.archive", plistlib.dumps(arch, fmt=plistlib.FMT_BINARY))
        for uuid, colour in (("COMP", (200, 0, 0, 255)), ("TOP", (0, 0, 255, 255))):
            for col in range((w + tile - 1) // tile):
                for row in range((h + tile - 1) // tile):
                    tw, th = min(tile, w - col * tile), min(tile, h - row * tile)
                    px = np.tile(np.array(colour, np.uint8), (th, tw, 1))
                    if uuid == "COMP" and row == 0:  # the bottom rows (row 0 = bottom) are green
                        px[:] = (0, 200, 0, 255)
                    z.writestr(f"{uuid}/{col}~{row}.chunk", lzo_literals(px.tobytes()))


def test_lzo_literal_and_match():
    assert procreate.lzo1x_decompress(lzo_literals(b"hello")) == b"hello"
    # literals "ab", then an M2 match copying 4 bytes from 2 back (overlapping): "ababab"
    src = bytes([17 + 2]) + b"ab" + bytes([((4 - 1) << 5) | (((2 - 1) & 7) << 2), (2 - 1) >> 3]) + b"\x11\x00\x00"
    assert procreate.lzo1x_decompress(src) == b"ababab"


def test_flattened_and_layers():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.procreate")
        make_file(p)
        img, doc = procreate.flattened(p)
        assert doc["name"] == "Tiny" and img.size == (6, 5)
        a = np.asarray(img)
        assert tuple(a[0, 0]) == (200, 0, 0) and tuple(a[-1, 0]) == (0, 200, 0)  # row 0 of the tiles is the bottom
        layers, _ = procreate.layers(p)
        assert [i["name"] for i, _ in layers] == ["Ink"]  # "Paint" has no tiles: left out
        assert layers[0][1].mode == "RGBA" and layers[0][1].getpixel((2, 2))[:3] == (0, 0, 255)


def test_real_file_if_present():
    p = "/mnt/user-data/uploads/Day_7_Vampire_.procreate"
    if not os.path.exists(p):
        return
    img, doc = procreate.flattened(p)
    assert img.size == (4000, 4000) and len(doc["layers"]) == 13
