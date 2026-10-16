"""Small JPEG thumbnails for the galleries.

Grids show 384-px thumbnails instead of full-size images, which keeps scrolling smooth and
memory low on older Macs with big libraries. Thumbnails live in ~/.protoartstudio/thumbs/<id>.jpg.
"""
from pathlib import Path

import paths

SIZE = 384
MAX_PER_CALL = 60  # create at most this many missing thumbnails per library query


def thumb_path(image_id):
    return paths.HOME / "thumbs" / f"{int(image_id)}.jpg"


def make(image_id, source, image=None):
    """Create the thumbnail; returns its path or None. `image` may be an already-open PIL image."""
    try:
        from PIL import Image
        dest = thumb_path(image_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im = image.copy() if image is not None else Image.open(source)
        im.thumbnail((SIZE, SIZE))
        im.convert("RGB").save(dest, "JPEG", quality=82)
        return dest
    except Exception:
        return None


def ensure(records):
    """Fill in record["thumb"], creating a few missing thumbnails each call."""
    made = 0
    for r in records:
        p = thumb_path(r["id"])
        if not p.exists() and made < MAX_PER_CALL and Path(r["path"]).exists():
            if make(r["id"], r["path"]):
                made += 1
        r["thumb"] = str(p) if p.exists() else None
    return records


def remove(ids):
    for i in ids:
        try:
            thumb_path(i).unlink()
        except OSError:
            pass
