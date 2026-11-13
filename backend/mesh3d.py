"""🧊 Make 3D: one picture -> a 3D model (.glb and .obj), with Tencent's Hunyuan3D 2 mini (0.6B shape model), the way
Modly (github.com/lightningpixel/modly, MIT) runs it on Apple Silicon. Tencent Hunyuan 3D 2.0 is under the Tencent
Hunyuan 3D 2.0 Community License (not for use in the EU, the UK or South Korea), so nothing of it ships with the app:
on first use, after you accept that licence, this Mac downloads
  * its code (hy3dgen, from a pinned commit of github.com/Tencent-Hunyuan/Hunyuan3D-2, ~1 MB),
  * a few small libraries it needs (einops, trimesh, scikit-image, pymeshlab…; ~150 MB, into the app's data folder;
    downloaded with curl from PyPI, then installed from those files), and
  * the model (tencent/Hunyuan3D-2mini, the base shape model only, ~2.5 GB, kept with the downloaded models).
It runs as a short-lived process (mesh3d_worker.py) so all its memory is given back afterwards.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
COMMIT = "f8db63096c8282cb27354314d896feba5ba6ff8a"
CODE_URL = f"https://codeload.github.com/Tencent-Hunyuan/Hunyuan3D-2/zip/{COMMIT}"
REPO = "tencent/Hunyuan3D-2mini"
SUBFOLDER = "hunyuan3d-dit-v2-mini"
# only what the app loads (the half-precision .safetensors and its config): the folder also holds a .ckpt copy
PATTERNS = [f"{SUBFOLDER}/config.yaml", f"{SUBFOLDER}/model.fp16.safetensors"]
# installed from wheels this Mac downloads with curl (pip's own connections are what a VPN or firewall blocks), without
# their dependencies: those the app already has (numpy, scipy, Pillow, packaging) come from the app, the rest are listed
LIBS = ["einops", "trimesh", "scikit-image", "pymeshlab", "networkx", "imageio", "tifffile", "lazy_loader"]
MODEL_GB, LIBS_GB = 2.5, 0.2
LICENSE_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE"
NOTICE = ("Tencent Hunyuan 3D 2.0 is licensed under the Tencent Hunyuan 3D 2.0 Community License Agreement, Copyright © "
          "2025 Tencent. All Rights Reserved. The trademark rights of “Tencent Hunyuan” are owned by Tencent or its "
          "affiliate. Protoart Studio is not affiliated with, sponsored or endorsed by Tencent.")


class Mesh3DError(Exception):
    pass


def root():
    return paths.HOME / "hunyuan3d"


def site():
    return root() / "site"


def code_dir():
    return root() / "code"


def _marker():
    return root() / ".ready"


def env_ready():
    return _marker().exists() and (code_dir() / "hy3dgen" / "shapegen" / "__init__.py").exists() and site().exists()


def model_downloaded():
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, allow_patterns=PATTERNS, local_files_only=True)
        return True
    except Exception:
        return False


def status():
    return {"env": env_ready(), "model": model_downloaded(), "model_gb": MODEL_GB, "libs_gb": LIBS_GB,
            "license_url": LICENSE_URL, "notice": NOTICE}


def _run(cmd, emit, cancel_event, what):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(HERE))
    tail, last = [], 0
    for line in p.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        print("[mesh3d-setup]", line, file=sys.stderr, flush=True)
        if cancel_event is not None and cancel_event.is_set():
            p.kill()
            from engine import Cancelled
            raise Cancelled()
        if line.startswith(("Collecting", "Downloading", "Installing")) and time.time() - last > 1:
            emit({"type": "status", "msg": f"{what}: " + line[:90]})
            last = time.time()
    if p.wait() != 0:
        raise Mesh3DError(f"{what} failed:\n" + "\n".join(tail[-8:]))


def _fetch(url, timeout=300):
    """A download: curl first (a VPN or firewall can block Python's own connections)."""
    import net
    if net.CURL and os.path.exists(net.CURL):
        r = subprocess.run([net.CURL, "-sS", "-L", "--fail", "--connect-timeout", "20", "--max-time", str(timeout),
                            "-A", "ProtoartStudio", url], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        print(f"[mesh3d] curl: {r.stderr.decode(errors='replace')[:200]}", flush=True)
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "ProtoartStudio"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _wheel_ok(name):
    """A wheel this Mac's Python can use: pure Python, or built for this Python on Apple Silicon macOS."""
    if not name.endswith(".whl"):
        return False
    tags = name[:-4].split("-")[-3:]  # python tag, abi tag, platform tag
    py, abi, plat = tags
    if plat == "any":
        return "py3" in py.split(".") or py.startswith("py3")
    v = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    if plat.startswith("macosx") and ("arm64" in plat or "universal2" in plat):
        return v in py.split(".") and (abi in (v, "abi3", "none")) or (abi == "abi3" and py.startswith("cp3")
                                                                        and int(py[3:] or 0) <= sys.version_info[1])
    return False


def _python_ok(spec):
    """The release says it works with this Python (requires_python), or says nothing."""
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        return f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}" in SpecifierSet(spec)
    except Exception:
        return True


def _pick_wheel(pkg):
    """(filename, url) of the newest release of pkg with a wheel for this Mac, from PyPI's JSON."""
    info = json.loads(_fetch(f"https://pypi.org/pypi/{pkg}/json", timeout=60))
    from packaging.version import InvalidVersion, Version

    def key(v):
        try:
            return Version(v)
        except InvalidVersion:
            return Version("0")
    for ver in sorted(info.get("releases", {}), key=key, reverse=True):
        if key(ver).is_prerelease:
            continue
        for f in info["releases"][ver]:
            if f.get("yanked"):
                continue
            if _wheel_ok(f["filename"]) and _python_ok(f.get("requires_python")):
                return f["filename"], f["url"], (f.get("digests") or {}).get("sha256")
    raise Mesh3DError(f"No {pkg} package for this Mac was found on PyPI.")


def setup(emit, cancel_event=None):
    """One-time: Hunyuan3D's code and the libraries it needs (the model downloads in the worker)."""
    if env_ready():
        return
    root().mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root()).free / 2**30
    if free < MODEL_GB + LIBS_GB + 2:
        raise Mesh3DError(f"Make 3D needs about {MODEL_GB + LIBS_GB:.0f} GB the first time; only {free:.1f} GB is free.")
    emit({"type": "status", "msg": "Make 3D: downloading Hunyuan3D's code (one time)…"})
    try:
        data = _fetch(CODE_URL)
        z = zipfile.ZipFile(io.BytesIO(data))
    except Exception as e:
        raise Mesh3DError(f"Couldn't download Hunyuan3D's code from GitHub ({e}). Check the internet connection.")
    shutil.rmtree(code_dir(), ignore_errors=True)
    for name in z.namelist():
        rel = name.split("/", 1)[-1]
        if not (rel.startswith("hy3dgen/shapegen/") or rel in ("hy3dgen/__init__.py", "LICENSE", "NOTICE")):
            continue
        if name.endswith("/") or ".." in rel:
            continue
        out = code_dir() / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(name))
    (code_dir() / "hy3dgen" / "__init__.py").touch()
    import hashlib
    wheels = root() / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    wheels.mkdir(parents=True)
    files = []
    for i, pkg in enumerate(LIBS, 1):
        if cancel_event is not None and cancel_event.is_set():
            from engine import Cancelled
            raise Cancelled()
        emit({"type": "status", "msg": f"Make 3D: downloading the libraries it needs ({i}/{len(LIBS)}: {pkg}, one time)…"})
        try:
            name, url, sha = _pick_wheel(pkg)
            data = _fetch(url)
        except Mesh3DError:
            raise
        except Exception as e:
            raise Mesh3DError(f"Couldn't download {pkg} ({e}). Check the internet connection.")
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise Mesh3DError(f"The {pkg} download was damaged; try again.")
        (wheels / name).write_bytes(data)
        files.append(str(wheels / name))
    emit({"type": "status", "msg": "Make 3D: installing the libraries (one time)…"})
    shutil.rmtree(site(), ignore_errors=True)
    # from the downloaded files only (no network), into their own folder that goes after the app's own libraries on
    # the path, so the app's numpy, scipy, torch… are the ones used
    _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--no-deps",
          "--target", str(site()), *files], emit, cancel_event, "Make 3D: installing")
    shutil.rmtree(wheels, ignore_errors=True)
    _marker().write_text(COMMIT)


def run(image_path, out_stem, opts, emit, cancel_event=None):
    """The worker turns a cut-out picture (RGBA) into out_stem.glb / .obj / _preview.png -> its result dict."""
    import net
    env = dict(os.environ, PYTHONUNBUFFERED="1", PROTOART_MESH3D_SITE=str(site()), PROTOART_MESH3D_CODE=str(code_dir()))
    if not model_downloaded():
        if net.strict:
            raise Mesh3DError(f"Make 3D needs a one-time download of Hunyuan3D 2 mini (~{MODEL_GB} GB), and Strict "
                              "offline mode is on. Turn it off in Settings → Privacy.")
        emit({"type": "status", "msg": f"Make 3D: downloading Hunyuan3D 2 mini (~{MODEL_GB} GB, first time only)…"})
        # here, not in the worker: this process's downloads go through curl when Python's own connections are blocked
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(REPO, allow_patterns=PATTERNS)
        except Exception as e:
            raise Mesh3DError(f"Couldn't download Hunyuan3D 2 mini ({e}). Check the internet connection and try again.")
    env["HF_HUB_OFFLINE"] = "1"
    # the downloaded folder itself (an absolute path): Hunyuan3D's loader then reads it as is. Given the repo name it
    # asks Hugging Face for the whole folder, .ckpt copy included, which we don't download ("snapshot is incomplete")
    from huggingface_hub import snapshot_download
    local = snapshot_download(REPO, allow_patterns=PATTERNS, local_files_only=True)
    args = dict(opts, image=str(image_path), out=str(out_stem), repo=str(Path(local).resolve()), subfolder=SUBFOLDER)
    p = subprocess.Popen([sys.executable, str(HERE / "mesh3d_worker.py"), json.dumps(args)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=str(HERE), env=env)
    tail = []

    def read_err():
        for line in p.stderr:
            tail.append(line.rstrip())
            del tail[:-40]
            print("[mesh3d]", line.rstrip(), file=sys.stderr, flush=True)

    threading.Thread(target=read_err, daemon=True).start()
    result = None
    for line in p.stdout:
        if cancel_event is not None and cancel_event.is_set():
            p.kill()
            from engine import Cancelled
            raise Cancelled()
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("type") in ("status", "progress"):
            emit(msg)
        elif msg.get("type") in ("done", "error"):
            result = msg
    p.wait()
    if not result or result.get("type") == "error":
        raise Mesh3DError((result or {}).get("msg") or "\n".join(tail[-6:]) or "Make 3D stopped")
    return result
