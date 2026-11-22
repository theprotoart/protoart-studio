"""🧩 Split a character sheet: figures are found (also when they touch), in reading order, and neighbours' bits are dropped."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import postprocess.sheet as sh  # noqa: E402


def _sheet(rows=3, cols=4, cell=100, touch=False):
    m = np.zeros((rows * cell, cols * cell), np.float32)
    for r in range(rows):
        for c in range(cols):
            y, x = r * cell, c * cell
            m[y + 15:y + 85, x + 25:x + 75] = 1  # a body
            m[y + 5:y + 30, x + 35:x + 65] = 1   # a head
            if touch and c < cols - 1:
                m[y + 60:y + 66, x + 75:x + 125] = 1  # an arm reaching into the next figure
    return m


def test_separate_figures():
    boxes = sh.find_figures(_sheet())
    assert len(boxes) == 12
    assert boxes[0][0] < boxes[1][0] and boxes[3][1] < boxes[4][1]  # left to right, then the next row


def test_touching_figures_are_parted():
    boxes = sh.find_figures(_sheet(touch=True))
    assert len(boxes) == 12


def test_grid():
    assert sh.grid_boxes(400, 300, 3, 4)[5] == (100, 100, 200, 200)


def test_neighbour_bits_dropped():
    a = np.zeros((120, 120), np.float32)
    a[20:100, 30:90] = 1      # the figure
    a[50:60, 0:12] = 1        # a neighbour's hand, cut off at the crop's left edge
    out = sh.keep_main(a, (10, 10, 110, 110))
    assert out[60, 60] == 1 and out[55, 5] == 0
    assert sh.tight_box(out, 0) == (30, 20, 90, 100)


def test_square():
    from PIL import Image
    sq = sh.on_square(Image.new("RGBA", (50, 100)))
    assert sq.width == sq.height


def test_touch_ups():
    import base64
    import io
    from PIL import Image
    import postprocess.cutout as cut
    a = np.zeros((40, 40), np.float32)
    a[10:30, 10:30] = 1
    f = Image.new("RGBA", (20, 20), (0, 0, 0, 0))  # painted at half size on the preview
    f.putpixel((2, 2), (255, 0, 0, 255))   # add: top-left (4..5, 4..5)
    f.putpixel((10, 10), (0, 0, 255, 255))  # remove: the middle
    buf = io.BytesIO()
    f.save(buf, "PNG")
    out = cut.apply_fix(a, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert out[4, 4] == 1 and out[20, 20] == 0 and out[15, 15] == 1 and out[35, 35] == 0


def test_touching_rows_are_parted():
    """every figure's raised hands touch the one above: the pieces are whole columns, the sheet's repeat parts them"""
    m = _sheet(rows=4, cols=3)
    for r in range(3):
        for c in range(3):
            y, x = r * 100, c * 100
            m[y + 80:y + 110, x + 30:x + 70] = 1  # overlapping into the next row
    boxes = sh.find_figures(m)
    assert len(boxes) == 12, len(boxes)


def test_neighbour_slice_trimmed():
    a = np.zeros((140, 100), np.float32)
    a[20:100, 20:80] = 1      # the figure (cell 10..110 tall)
    a[100:140, 40:60] = 1     # the neighbour below, joined to it, running on to the crop's bottom edge
    a[15:25, 80:90] = 1       # the figure's own hand sticking out of the cell, not reaching the edge
    out = sh.keep_main(a, (10, 10, 90, 110))
    assert out[130, 50] == 0 and out[60, 50] == 1 and out[20, 85] == 1
