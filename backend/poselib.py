"""🧍 Pose library: ready-made pose pictures for Create's 🎛 ControlNet, from Shisei (https://github.com/wquguru/shisei),
a free (CC0) library of mannequin poses, each drawn from up to five angles as line art, with depth maps and some photo
renders. The list of poses ships with the app (data/shisei_poses.json: the poses Shisei rates "general"); the pictures
download from GitHub when first shown or used (a pinned commit, so they can't change) and are kept on this Mac.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "data" / "shisei_poses.json"
KINDS = ("line", "photo", "depth")
VIEWS = ("front", "diag", "side", "above", "below")
_cat = None


def catalog():
    global _cat
    if _cat is None:
        _cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    return _cat


MINE_CAT = "★ My poses"


def _mine_dir():
    d = paths.HOME / "poselib" / "mine"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mine():
    """Your own pose pictures (added in the Pose library; copies kept in Protoart's data folder)."""
    out = []
    for f in sorted(_mine_dir().glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append({"id": "mine:" + f.name, "series": "mine", "cat": MINE_CAT, "name": f.stem.replace("_", " "),
                    "desc": "", "views": {"front": {"mine": f.name}}})
    return out


def _hidden_file():
    return paths.HOME / "poselib" / "hidden.json"


def hidden():
    try:
        return set(json.loads(_hidden_file().read_text()))
    except Exception:
        return set()


def set_hidden(pid, hide):
    """✕ on one of the library's poses hides it from the picker (↺ Show hidden brings it back)."""
    h = hidden()
    (h.add if hide else h.discard)(str(pid))
    _hidden_file().parent.mkdir(parents=True, exist_ok=True)
    _hidden_file().write_text(json.dumps(sorted(h)))


def listing():
    """Your own poses first, then the library's (the ones you hid are marked)."""
    c, h = catalog(), hidden()
    return dict(c, poses=mine() + [dict(p, hidden=True) if p["id"] in h else p for p in c["poses"]])


def add_mine(files):
    """Copies of pictures you pick (any picture the app can open) -> how many were added."""
    from PIL import Image, ImageOps
    import pictures
    n = 0
    for src in files[:50]:
        try:
            im = ImageOps.exif_transpose(Image.open(pictures.prepare(src))).convert("RGB")
        except Exception as e:
            print(f"[poselib] couldn't add {src}: {e}", flush=True)
            continue
        im.thumbnail((1536, 1536))
        stem = re.sub(r"[^\w\- ]+", "_", Path(src).stem).strip()[:60] or "pose"
        dest, k = _mine_dir() / f"{stem}.png", 2
        while dest.exists():
            dest, k = _mine_dir() / f"{stem} {k}.png", k + 1
        im.save(dest)
        n += 1
    return n


def remove_mine(pid):
    """Your own pose: its copy here is deleted (your original picture stays). A library pose: hidden."""
    if not str(pid).startswith("mine:"):
        return set_hidden(pid, True)
    name = str(pid).split(":", 1)[-1]
    if "/" in name or not name.endswith(".png"):
        raise ValueError("Only your own poses can be removed.")
    for f in (_mine_dir() / name, _cache() / "w300" / "mine" / name):
        try:
            f.unlink()
        except FileNotFoundError:
            pass


def _cache():
    d = paths.HOME / "poselib"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pose(pid):
    for p in (mine() if str(pid).startswith("mine:") else catalog()["poses"]):
        if p["id"] == pid:
            return p
    raise ValueError("That pose isn't in the library.")


def _fetch(kind, width, name):
    """One picture (downloaded once, then from this Mac) -> local path."""
    if kind not in KINDS or width not in (300, 900) or not re.match(r"^[\w.-]+\.webp$", name):
        raise ValueError("Not a pose library picture.")
    dest = _cache() / f"w{width}" / kind / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/wquguru/shisei/{catalog()['commit']}/images/{kind}/w{width}/{name}"
    import net
    data = None
    tmp = dest.with_suffix(".part")
    # curl first: on some Macs (a VPN, a firewall) Python's own connections are cut ("Errno 53, connection abort")
    # while curl gets through; the app downloads models the same way
    if net.CURL and os.path.exists(net.CURL):
        import subprocess
        r = subprocess.run([net.CURL, "-sS", "-L", "--fail", "--connect-timeout", "20", "--max-time", "90",
                            "-A", "ProtoartStudio", "-o", str(tmp), url], capture_output=True, text=True)
        if r.returncode == 0 and tmp.exists():
            data = tmp.read_bytes()
        else:
            print(f"[poselib] curl: {r.stderr.strip()[:200]}", flush=True)
    if data is None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ProtoartStudio"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read(8 * 1024 * 1024)
        except Exception as e:
            raise ValueError(f"Couldn't download the pose picture from GitHub ({e}). Check the internet connection "
                             "(a VPN or firewall can block it).")
    if not data.startswith(b"RIFF") or b"WEBP" not in data[:16]:
        raise ValueError("The pose picture didn't download properly; try again.")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def thumbs(ids, view="front"):
    """Small line drawings for the picker -> {id: png path}; poses whose picture can't be fetched are left out.
    Several download at once (each is ~20 KB)."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def one(pid):
        p = _pose(pid)
        v = p["views"].get(view) or next(iter(p["views"].values()))
        if v.get("mine"):  # your own: a small copy for the picker
            png = _cache() / "w300" / "mine" / v["mine"]
            if not png.exists():
                png.parent.mkdir(parents=True, exist_ok=True)
                im = Image.open(_mine_dir() / v["mine"]).convert("RGB")
                im.thumbnail((300, 400))
                im.save(png)
            return str(png)
        name = v.get("line") or v.get("photo") or v.get("depth")
        kind = "line" if v.get("line") else "photo" if v.get("photo") else "depth"
        src = _fetch(kind, 300, name)
        png = src.with_suffix(".png")
        if not png.exists():
            Image.open(src).convert("RGB").save(png)
        return str(png)

    out, errors = {}, []

    def safe(pid):
        try:
            out[pid] = one(pid)
        except Exception as e:
            errors.append(str(e))
            print(f"[poselib] thumbnail {pid}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(safe, [str(i) for i in ids[:200]]))
    if errors and not out:  # nothing at all came: say why (no internet, a VPN or firewall in the way)
        raise ValueError(errors[0])
    return out


def get(pid, view, kind):
    """The full-size picture to follow -> {"path": png, "already_map": bool}. kind: "pose" (the photo render, or the
    line drawing, for OpenPose to read), "depth" (its depth map, used as it is) or "line" (the line drawing)."""
    from PIL import Image
    p = _pose(pid)
    v = p["views"].get(view)
    if not v:
        raise ValueError("That angle isn't there for this pose.")
    if v.get("mine"):  # your own picture: OpenPose or the depth finder reads it like any other
        return {"path": str(_mine_dir() / v["mine"]), "already_map": False, "name": p["name"], "desc": ""}
    if kind == "depth":
        if not v.get("depth"):
            raise ValueError("This angle has no depth map: use it for 🦴 Pose instead.")
        src, already = _fetch("depth", 900, v["depth"]), True
    elif kind == "line":
        if not v.get("line"):
            raise ValueError("This angle has no line drawing.")
        src, already = _fetch("line", 900, v["line"]), False
    else:
        pick = "photo" if v.get("photo") else "line" if v.get("line") else None
        if not pick:
            raise ValueError("This angle has only a depth map: use it for 🧊 Depth instead.")
        src, already = _fetch(pick, 900, v[pick]), False
    out = _cache() / "use" / f"{pid}_{view}_{kind}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        Image.open(src).convert("RGB").save(out)
    return {"path": str(out), "already_map": already, "name": p["name"], "desc": p.get("desc") or ""}
