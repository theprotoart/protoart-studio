"""Object selection (Image to image and Tools → painting): you drag a box around something and its outline is found
with OpenCV's GrabCut (it separates the object from the background inside the box by colour and texture). No model,
no download, about a second. Returns a mask the size of the painting area."""
import base64
import io

import numpy as np
from PIL import Image


def select(image, rect, width, height):
    """image: PIL; rect: (x, y, w, h) as fractions of the picture; width/height: the painting area's size.
    Returns (data_url, found): a PNG that's pink where the object is and transparent elsewhere."""
    import cv2
    OW, OH = int(width), int(height)
    f = min(1.0, 512 / max(OW, OH))  # GrabCut works at up to 512 px (a second or less), the mask is scaled back up
    img = image.convert("RGB").resize((max(8, round(OW * f)), max(8, round(OH * f))), Image.LANCZOS)
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    W, H = img.size
    x = int(max(0, min(W - 2, rect[0] * W)))
    y = int(max(0, min(H - 2, rect[1] * H)))
    w = int(max(2, min(W - x, rect[2] * W)))
    h = int(max(2, min(H - y, rect[3] * H)))
    mask = np.zeros((H, W), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    found = True
    try:
        cv2.grabCut(arr, mask, (x, y, w, h), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = np.zeros((H, W), np.uint8)
    # tidy: close small holes, drop specks, keep the pieces that matter (at least 2% of the box)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.02 * w * h:
            keep[labels == i] = 1
    if not keep.any():
        found = False
    if (W, H) != (OW, OH):  # back to the painting area's size, with smooth edges
        keep = (cv2.resize(keep.astype(np.float32), (OW, OH), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
    rgba = np.zeros((OH, OW, 4), np.uint8)
    rgba[keep == 1] = (255, 64, 129, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), found
