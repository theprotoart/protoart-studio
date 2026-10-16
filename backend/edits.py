"""Round-trip editing in an outside image editor (PhotoCraft by default, or Photoshop, Pixelmator…).

"Edit in PhotoCraft" copies the picture into ~/.protoartstudio/edits/ and the app opens that copy in the
editor. Whenever the editor saves it (or saves/exports another picture into that folder), the new version
is added to the Library as an "Edited" picture linked to the original. The original is never changed.
"""
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import paths

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".psd", ".bmp", ".heic", ".heif", ".avif", ".gif"}


def folder():
    d = paths.HOME / "edits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path():
    return folder() / ".edits.json"


def _load():
    try:
        return json.loads(_state_path().read_text())
    except Exception:
        return {}


def _save(state):
    tmp = _state_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(_state_path())


def start(source_path, source_id=None, settings=None, editor="PhotoCraft"):
    """Make the working copy. -> path of the copy to open in the editor."""
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"The picture is missing: {src}")
    stem = f"{source_id}_" if source_id is not None else ""
    name = f"{stem}{src.stem[:60]}_edit_{datetime.now().strftime('%H%M%S')}.png"
    dest = folder() / name
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dest)
    else:
        from PIL import Image
        with Image.open(src) as im:
            im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(dest)
    st = dest.stat()
    state = _load()
    state[dest.name] = {"source_id": source_id, "source_path": str(src), "mtime": st.st_mtime, "size": st.st_size,
                        "editor": editor, "started": time.time(), "imported": 0,
                        "settings": settings or {}}
    _save(state)
    return str(dest)


def pending():
    """Working copies still being watched."""
    state = _load()
    return [{"file": str(folder() / k), "name": k, "editor": v.get("editor"), "source_id": v.get("source_id"),
             "imported": v.get("imported", 0)} for k, v in state.items() if not v.get("done") and not v.get("extra")]


def finish(name=None):
    """Stop watching one working copy (or all)."""
    state = _load()
    for k, v in state.items():
        if name is None or k == name:
            v["done"] = True
    _save(state)
    return pending()


def check(import_fn):
    """Find saved edits and new pictures in the edits folder; import_fn(path, info) adds one to the Library.
    -> list of new library records"""
    state = _load()
    records = []
    changed = False
    d = folder()
    now = time.time()
    for f in sorted(d.iterdir()):
        if f.name.startswith(".") or f.suffix.lower() not in IMAGE_EXT or not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if now - st.st_mtime < 1.5:
            continue  # still being written: look again next time
        info = state.get(f.name)
        if info is None:
            # a new file (e.g. "Save As" / export in the editor): link it to the copy it came from
            base = next((k for k in state if f.stem.startswith(Path(k).stem)), None)
            if base is None:
                prefix = f.stem.split("_", 1)[0]
                base = next((k for k in state if k.split("_", 1)[0] == prefix and prefix.isdigit()), None)
            parent = state.get(base, {}) if base else {}
            info = {"source_id": parent.get("source_id"), "source_path": parent.get("source_path"),
                    "editor": parent.get("editor", "editor"), "settings": parent.get("settings", {}),
                    "mtime": 0, "size": -1, "imported": 0, "started": now, "extra": True}
            state[f.name] = info
            changed = True
        if info.get("done"):
            continue
        if st.st_mtime > info.get("mtime", 0) + 0.5 or (st.st_size != info.get("size") and info.get("mtime", 0) == 0):
            try:
                rec = import_fn(f, info)
            except Exception as e:
                print(f"[edits] couldn't import {f.name}: {e}", flush=True)
                rec = None
            info["mtime"], info["size"] = st.st_mtime, st.st_size
            changed = True
            if rec:
                info["imported"] = info.get("imported", 0) + 1
                records.append(rec)
    if changed:
        _save(state)
    return records


def import_into_library(library, images_dir, thumbs, f, info, emit=None, extra=None, suffix="edited"):
    """Copy a saved edit into the image folder and add it to the Library, linked to its original."""
    from PIL import Image
    import pictures
    with Image.open(pictures.prepare(str(f))) as im:  # PSD, TIFF, HEIC… saved by an editor: read through macOS
        im.load()
        img = im.convert("RGBA" if im.mode in ("RGBA", "LA") else "RGB")
    day = Path(images_dir) / datetime.now().strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    stem = Path(f).stem[:60] if suffix == "edited" else Path(info.get("source_path") or f).stem[:60]
    dest = day / f"{datetime.now().strftime('%H%M%S')}_{stem}_{suffix}.png"
    n = 2
    while dest.exists():
        dest = dest.with_name(f"{dest.stem.rsplit('~', 1)[0]}~{n}.png")
        n += 1
    img.save(dest)
    src_settings = dict(info.get("settings") or {})
    src = None
    if info.get("source_id") is not None:
        try:
            src = library.get_image(int(info["source_id"]))
        except Exception:
            src = None
    if src:
        src_settings = dict(src.get("settings") or {}, **{k: src[k] for k in ("prompt", "negative_prompt", "model",
                                                                               "sampler", "steps", "cfg", "seed")})
        src_settings["loras"] = src.get("loras") or []
    settings = dict(src_settings, width=img.width, height=img.height, edited_in=info.get("editor") or "editor",
                    edited_from=info.get("source_path"))
    settings.setdefault("loras", [])
    if extra:
        settings.update(extra)
    record = library.add_image(dest, settings, parent_id=src["id"] if src else None, kind="edited")
    record["thumb"] = str(thumbs.make(record["id"], dest, img) or "") or None
    if emit:
        emit({"type": "image", "record": record})
    return record
