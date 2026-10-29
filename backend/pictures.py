"""Open (almost) any picture: iPhone HEIC/HEIF, TIFF, AVIF, PSD, JPEG 2000, camera RAW (DNG, CR2/CR3, NEF, ARW,
RAF, ORF, RW2…), BMP, GIF and more, besides PNG / JPEG / WebP.

PNG, JPEG, WebP, GIF and BMP are read directly. Everything else is converted once to a PNG by macOS's own image
reader (`/usr/bin/sips`, Apple's ImageIO, which macOS updates keep patched); no extra decoding libraries are added
to the app (libheif in particular, which has had serious security bugs). Conversions are cached in
~/.protoartstudio/imports, keyed by the file and its modification time, and capped at 8192 px on the long side.
"""
import hashlib
import os
import subprocess
from pathlib import Path

import paths

DIRECT = {".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".gif", ".bmp"}
VIA_MACOS = {".heic", ".heif", ".hif", ".avif", ".tif", ".tiff", ".psd", ".jp2", ".j2k", ".jpx", ".tga", ".ico",
             ".icns", ".exr", ".hdr", ".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
             ".orf", ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".erf", ".kdc", ".mrw", ".raw", ".rwl", ".iiq"}
SUPPORTED = DIRECT | VIA_MACOS | {".procreate"}
MAX_BYTES = 600 * 1024 * 1024
MAX_SIDE = 8192
SIPS = "/usr/bin/sips"


class PictureError(ValueError):
    pass


def imports_dir():
    d = paths.HOME / "imports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_supported(path):
    return Path(str(path)).suffix.lower() in SUPPORTED


def prepare(path):
    """-> a path PNG/JPEG/WebP/GIF/BMP readers can open: the file itself, or a converted PNG copy."""
    p = Path(str(path or ""))
    if not p.is_absolute() or not p.is_file():
        raise PictureError("That picture wasn't found.")
    ext = p.suffix.lower()
    if ext in DIRECT:
        return str(p)
    if ext == ".procreate":  # Procreate: its own flattened picture, read from inside the file
        return _from_procreate(p)
    if ext not in VIA_MACOS:
        raise PictureError(f"{p.suffix or 'That'} files aren't pictures this app can open. "
                           "PNG, JPEG, WebP, HEIC, TIFF, AVIF, PSD, GIF, BMP and camera RAW files work.")
    st = p.stat()
    if st.st_size > MAX_BYTES:
        raise PictureError("That picture is too big to open (over 600 MB).")
    key = hashlib.sha256(f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:24]
    out = imports_dir() / f"{p.stem[:40]}_{key}.png"
    if out.is_file() and out.stat().st_size > 0:
        return str(out)
    if not os.path.exists(SIPS):
        raise PictureError(f"{p.suffix.upper()[1:]} pictures need macOS's own image reader; convert it to PNG or JPEG first.")
    tmp = out.with_name(out.stem + ".part.png")
    try:
        r = subprocess.run([SIPS, "-s", "format", "png", "-Z", str(MAX_SIDE), str(p), "--out", str(tmp)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            raise PictureError(f"macOS couldn't read {p.name}: it may be damaged, or a kind of {p.suffix.upper()[1:]} "
                               "it doesn't support.")
        os.replace(tmp, out)
    except subprocess.TimeoutExpired:
        raise PictureError(f"Opening {p.name} took too long.")
    finally:
        if tmp.exists():
            tmp.unlink()
    return str(out)


def _from_procreate(p):
    """A .procreate file -> a PNG of the finished picture (cached like the other conversions)."""
    import procreate
    st = p.stat()
    if st.st_size > MAX_BYTES * 3:
        raise PictureError("That Procreate file is too big to open here.")
    key = hashlib.sha256(f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:24]
    out = imports_dir() / f"{p.stem[:40]}_{key}.png"
    if out.is_file() and out.stat().st_size > 0:
        return str(out)
    try:
        img, _doc = procreate.flattened(str(p))
    except procreate.ProcreateError as e:
        raise PictureError(str(e))
    except Exception:
        raise PictureError(f"Couldn't read {p.name}: it may be damaged, or from a Procreate version this app doesn't know.")
    tmp = out.with_name(out.stem + ".part.png")
    img.save(tmp)
    os.replace(tmp, out)
    return str(out)


def clear_cache(max_files=200):
    """Keep the imports folder small: drop the oldest conversions."""
    try:
        files = sorted(imports_dir().glob("*.png"), key=lambda f: f.stat().st_mtime)
        for f in files[:-max_files] if len(files) > max_files else []:
            f.unlink()
    except OSError:
        pass
