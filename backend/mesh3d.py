"""🧊 Make 3D: one picture -> a 3D model (.glb and .obj), with Tencent's Hunyuan3D 2 mini (0.6B shape model), the way
Modly (github.com/lightningpixel/modly, MIT) runs it on Apple Silicon. Tencent Hunyuan 3D 2.0 is under the Tencent
Hunyuan 3D 2.0 Community License (not for use in the EU, the UK or South Korea), so nothing of it ships with the app:
on first use, after you accept that licence, this Mac downloads
  * its code (hy3dgen, from a pinned commit of github.com/Tencent-Hunyuan/Hunyuan3D-2, ~1 MB),
  * a few small libraries it needs (einops, trimesh, scikit-image, pymeshlab; ~150 MB, into the app's data folder), and
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
LIBS = ["einops>=0.8", "trimesh>=4.4", "scikit-image>=0.24", "pymeshlab>=2023.12"]
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
        snapshot_download(REPO, allow_patterns=[f"{SUBFOLDER}/*"], local_files_only=True)
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


def _fetch(url):
    """A small download (the code): curl first (a VPN or firewall can block Python's own connections)."""
    import net
    if net.CURL and os.path.exists(net.CURL):
        r = subprocess.run([net.CURL, "-sS", "-L", "--fail", "--connect-timeout", "20", "--max-time", "300",
                            "-A", "ProtoartStudio", url], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        print(f"[mesh3d] curl: {r.stderr.decode(errors='replace')[:200]}", flush=True)
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "ProtoartStudio"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


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
    emit({"type": "status", "msg": "Make 3D: installing the libraries it needs (one time, ~150 MB)…"})
    shutil.rmtree(site(), ignore_errors=True)
    # into their own folder, after the app's own libraries on the path (so the app's numpy, torch… are the ones used)
    _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--target", str(site()), *LIBS],
         emit, cancel_event, "Make 3D: installing")
    _marker().write_text(COMMIT)


def run(image_path, out_stem, opts, emit, cancel_event=None):
    """The worker turns a cut-out picture (RGBA) into out_stem.glb / .obj / _preview.png -> its result dict."""
    import net
    env = dict(os.environ, PYTHONUNBUFFERED="1", PROTOART_MESH3D_SITE=str(site()), PROTOART_MESH3D_CODE=str(code_dir()))
    if not model_downloaded():
        if net.strict:
            raise Mesh3DError(f"Make 3D needs a one-time download of Hunyuan3D 2 mini (~{MODEL_GB} GB), and Strict "
                              "offline mode is on. Turn it off in Settings → Privacy.")
        env["HF_HUB_OFFLINE"] = "0"
        emit({"type": "status", "msg": f"Make 3D: downloading Hunyuan3D 2 mini (~{MODEL_GB} GB, first time only)…"})
    else:
        env["HF_HUB_OFFLINE"] = "1"
    args = dict(opts, image=str(image_path), out=str(out_stem), repo=REPO, subfolder=SUBFOLDER)
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
