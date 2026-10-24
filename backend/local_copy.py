"""Models → right-click → 💾 Keep a copy on this Mac.

A model file on an external drive takes 60–90 s to load; the same file on the Mac's own disk takes 3–10 s. This
copies a .safetensors checkpoint from a drive into Protoart's own models folder, where it's listed as one of
"your checkpoints", and from then on the copy is the one listed and used (the drive's file is hidden from the model
lists while the copy is there). The file on the drive is only read: never changed, moved or deleted. Move the copy
to the Trash (right-click → Show in Finder) and the drive's file is listed again.
"""
import os
import shutil
from pathlib import Path

import paths

KEPT = {}            # saved setting "kept_copies": original file path -> name of the copy in paths.MODELS_DIR
MARGIN = 3 * 10**9   # always leave this much free on the Mac (macOS needs room for memory swap on 8 GB Macs)
SIDECARS = (".civitai.info", ".preview.png", ".preview.jpg", ".png", ".jpg", ".jpeg", ".webp", ".json", ".txt", ".yaml")
CHUNK = 16 * 2**20


def on_external_drive(path):
    return os.path.realpath(str(path)).startswith("/Volumes/")


def target_for(src):
    return paths.MODELS_DIR / Path(src).name


def copy_id(src):
    """The model id of a kept copy, or None (no copy, or it has been moved to the Trash)."""
    name = KEPT.get(os.path.realpath(str(src))) or KEPT.get(str(src))
    return ("file:" + name) if name and (paths.MODELS_DIR / name).is_file() else None


def check(src):
    """Can this file be copied? -> {"ok", "why", "size", "free", "need", "target"}. Nothing is written."""
    p = Path(str(src or ""))
    out = {"ok": False, "why": "", "size": 0, "free": 0, "need": 0, "target": ""}
    if not p.is_file():
        out["why"] = "The file isn't there: is its drive connected?"
        return out
    if p.suffix.lower() != ".safetensors":
        out["why"] = "Only single .safetensors model files can be kept on this Mac (convert a .ckpt first)."
        return out
    if not on_external_drive(p):
        out["why"] = "It's already on this Mac's own disk."
        return out
    size = p.stat().st_size
    dest = target_for(p)
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.MODELS_DIR).free
    out.update(size=size, free=free, need=size + MARGIN, target=str(dest))
    if dest.exists():
        if dest.stat().st_size == size:
            out.update(ok=True, why="already copied")  # the same file is already there: just use it
            return out
        out["why"] = f"A different file called {dest.name} is already in Protoart's models folder."
        return out
    if free < size + MARGIN:
        out["why"] = (f"Not enough free space on this Mac: it needs {size / 1e9:.1f} GB plus "
                      f"{MARGIN / 1e9:.0f} GB to spare, and {free / 1e9:.1f} GB is free.")
        return out
    out["ok"] = True
    return out


def _sidecars(src):
    folder, stem = src.parent, src.stem
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return [folder / n for n in names
            if n.startswith(stem + ".") and n != src.name and n.lower().endswith(SIDECARS) and not n.startswith("._")]


def copy(src, progress=None, cancelled=None):
    """Copy src (and its info / preview files) into Protoart's models folder. -> {"id", "path", "gb", "existed"}"""
    progress = progress or (lambda done, total: None)
    cancelled = cancelled or (lambda: False)
    src = Path(str(src))
    c = check(src)
    if not c["ok"]:
        raise ValueError(c["why"])
    dest = Path(c["target"])
    existed = dest.exists()
    if not existed:
        tmp = dest.with_name(dest.name + ".part")
        total, done = c["size"], 0
        try:
            with open(src, "rb") as fi, open(tmp, "wb") as fo:  # the original is only ever opened for reading
                while True:
                    if cancelled():
                        raise InterruptedError()
                    b = fi.read(CHUNK)
                    if not b:
                        break
                    fo.write(b)
                    done += len(b)
                    progress(done, total)
                fo.flush()
                os.fsync(fo.fileno())
            if tmp.stat().st_size != total:
                raise ValueError("The copy came out a different size from the original (was the drive unplugged?). "
                                 "Nothing was kept; try again.")
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
    for s in _sidecars(src):  # civitai info (keeps its VAE), preview pictures, notes: small, never overwritten
        d = paths.MODELS_DIR / s.name
        if not d.exists():
            try:
                shutil.copyfile(s, d)
            except OSError:
                pass
    KEPT[os.path.realpath(str(src))] = dest.name
    return {"id": "file:" + dest.name, "path": str(dest), "gb": round(c["size"] / 1e9, 1), "existed": existed}
