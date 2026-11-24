"""🧊 Make 3D: one picture -> a 3D Gaussian splat (.ply), with Speridlabs' Idol (github.com/speridlabs/idol-sdk,
Apache-2.0). Only for Macs with 16 GB or more: it peaks at about 7 GB on a graphics card, more on a Mac's shared memory.
Nothing of it ships with the app: on first use this Mac downloads
  * Idol's code (from a pinned commit of github.com/speridlabs/idol-sdk, ~100 KB), plus plyfile (a small library),
  * Idol's weights (speridlabs/idol, ~3 GB, Apache-2.0),
  * the FLUX.2 image encoder (only the VAE of black-forest-labs/FLUX.2-klein-4B, Apache-2.0, ~170 MB), and
  * Meta's DINOv3 image encoder (facebook/dinov3-vith16plus-pretrain-lvd1689m, ~1.7 GB). It is gated: you accept
    Meta's DINOv3 licence on Hugging Face once, and save a Hugging Face access token in the app.
It runs as a short-lived process (idol_worker.py) so all its memory is given back afterwards. Idol uses tensordict for
a few simple dictionaries of tensors; a small stand-in (SHIM below) is used instead of that compiled library, which
would have to match the app's PyTorch exactly.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
COMMIT = "8db700349a5570aac390e4780fa32e3eed6b587a"
CODE_URL = f"https://codeload.github.com/speridlabs/idol-sdk/zip/{COMMIT}"
WEIGHTS = "speridlabs/idol"
WEIGHT_FILES = ["config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors",
                "birefnet.safetensors"]
DINOV3 = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
FLUX2_VAE = "black-forest-labs/FLUX.2-klein-4B"  # the same autoencoder as FLUX.2-dev's, and not gated
LIBS = ["plyfile"]
MIN_RAM_GB = 15.5          # a "16 GB" Mac reports a little under 16
BIG_RAM_GB = 23.5          # 24 GB and more: the full-size settings
DOWNLOAD_GB = 5.0
LICENSE_URL = "https://github.com/speridlabs/idol-sdk/blob/main/LICENSE"
DINOV3_PAGE = f"huggingface.co/{DINOV3}"
NOTICE = ("Idol © Speridlabs, Apache License 2.0. DINOv3 © Meta Platforms, under the DINOv3 License. FLUX.2 [klein] "
          "autoencoder © Black Forest Labs, Apache License 2.0. Protoart Studio is not affiliated with them.")

# what each quality choice does: (steps, Gaussians) on a 16 GB Mac and on 24 GB or more
QUALITY = {"quick": ((10, 65536), (10, 65536)),
           "standard": ((20, 65536), (25, 131072)),
           "best": ((30, 65536), (50, 262144))}

SHIM = '''"""A small stand-in for tensordict.TensorDict, as much of it as Idol uses (written by Protoart Studio)."""
import torch


class TensorDict(dict):
    def __init__(self, source=None, batch_size=None, device=None, **kw):
        super().__init__(source or {})
        if isinstance(batch_size, int):
            batch_size = [batch_size]
        self.batch_size = torch.Size(batch_size or [])

    @property
    def shape(self):
        return self.batch_size

    def __getitem__(self, key):
        if isinstance(key, str):
            return dict.__getitem__(self, key)
        out = {k: v[key] for k, v in self.items()}
        n = next(iter(out.values())).shape[0] if out else 0
        return TensorDict(out, batch_size=[n])

    def apply(self, fn):
        return TensorDict({k: fn(v) for k, v in self.items()}, batch_size=self.batch_size)

    def _op(self, other, fn):
        if isinstance(other, TensorDict):
            return TensorDict({k: fn(v, other[k]) for k, v in self.items()}, batch_size=self.batch_size)
        return TensorDict({k: fn(v, other) for k, v in self.items()}, batch_size=self.batch_size)

    def __add__(self, o): return self._op(o, lambda a, b: a + b)
    def __radd__(self, o): return self._op(o, lambda a, b: b + a)
    def __sub__(self, o): return self._op(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._op(o, lambda a, b: b - a)
    def __mul__(self, o): return self._op(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._op(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._op(o, lambda a, b: a / b)
    def __neg__(self): return self.apply(lambda v: -v)
    __array_priority__ = 1000  # numpy floats on the left hand over to __rmul__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.cat and args and isinstance(args[0], (list, tuple)) and isinstance(args[0][0], TensorDict):
            parts = args[0]
            out = {k: torch.cat([p[k] for p in parts], *args[1:], **(kwargs or {})) for k in parts[0]}
            return TensorDict(out, batch_size=[sum(p.batch_size[0] for p in parts)])
        return NotImplemented

    def to(self, *a, **kw):
        return TensorDict({k: v.to(*a, **kw) for k, v in self.items()}, batch_size=self.batch_size)
'''


class IdolError(Exception):
    pass


def root():
    return paths.HOME / "idol"


def site():
    return root() / "site"


def code_dir():
    return root() / "code"


def _marker():
    return root() / ".ready"


def ram_gb():
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip()) / 2**30
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except Exception:
            return 0.0


def big_mac(ram=None):
    return (ram if ram is not None else ram_gb()) >= BIG_RAM_GB


def settings_for(quality, ram=None):
    """(steps, number of Gaussians) for a quality choice on this Mac."""
    pick = QUALITY.get(quality) or QUALITY["standard"]
    return pick[1] if big_mac(ram) else pick[0]


def env_ready():
    return (_marker().exists() and (code_dir() / "idol" / "pipeline.py").exists()
            and (site() / "tensordict" / "__init__.py").exists())


def _cached(repo, patterns):
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
    except Exception:
        return None


def models_downloaded():
    return {"weights": bool(_cached(WEIGHTS, WEIGHT_FILES)), "dinov3": bool(_cached(DINOV3, ["*.json", "*.safetensors"])),
            "vae": bool(_cached(FLUX2_VAE, ["vae/*"]))}


def status():
    ram = ram_gb()
    return {"env": env_ready(), "models": models_downloaded(), "ram_gb": round(ram, 1), "ram_ok": ram >= MIN_RAM_GB,
            "big": big_mac(ram), "download_gb": DOWNLOAD_GB, "token": bool(os.environ.get("HF_TOKEN")),
            "dinov3_page": DINOV3_PAGE, "license_url": LICENSE_URL, "notice": NOTICE,
            "quality": {k: dict(zip(("steps", "gaussians"), settings_for(k, ram))) for k in QUALITY}}


def _run(cmd, emit, cancel_event, what):
    import mesh3d
    return mesh3d._run(cmd, emit, cancel_event, what)


def setup(emit, cancel_event=None):
    """One-time: Idol's code, plyfile and the tensordict stand-in (the models download in run())."""
    if env_ready():
        return
    import hashlib

    import mesh3d
    root().mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root()).free / 2**30
    if free < DOWNLOAD_GB + 2:
        raise IdolError(f"Make 3D needs about {DOWNLOAD_GB:.0f} GB the first time; only {free:.1f} GB is free.")
    emit({"type": "status", "msg": "Make 3D: downloading Idol's code (one time)…"})
    try:
        z = zipfile.ZipFile(io.BytesIO(mesh3d._fetch(CODE_URL)))
    except Exception as e:
        raise IdolError(f"Couldn't download Idol's code from GitHub ({e}). Check the internet connection.")
    shutil.rmtree(code_dir(), ignore_errors=True)
    for name in z.namelist():
        rel = name.split("/", 1)[-1]
        if not (rel.startswith("idol/") or rel == "LICENSE") or name.endswith("/") or ".." in rel:
            continue
        out = code_dir() / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(name))
    if not (code_dir() / "idol" / "pipeline.py").exists():
        raise IdolError("Idol's code download was incomplete; try again.")
    wheels = root() / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    wheels.mkdir(parents=True)
    files = []
    for pkg in LIBS:
        if cancel_event is not None and cancel_event.is_set():
            from engine import Cancelled
            raise Cancelled()
        emit({"type": "status", "msg": f"Make 3D: downloading {pkg} (one time)…"})
        try:
            name, url, sha = mesh3d._pick_wheel(pkg)
            data = mesh3d._fetch(url)
        except Exception as e:
            raise IdolError(f"Couldn't download {pkg} ({e}). Check the internet connection.")
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise IdolError(f"The {pkg} download was damaged; try again.")
        (wheels / name).write_bytes(data)
        files.append(str(wheels / name))
    shutil.rmtree(site(), ignore_errors=True)
    emit({"type": "status", "msg": "Make 3D: installing (one time)…"})
    _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--no-deps",
          "--target", str(site()), *files], emit, cancel_event, "Make 3D: installing")
    shutil.rmtree(wheels, ignore_errors=True)
    (site() / "tensordict").mkdir(parents=True, exist_ok=True)
    (site() / "tensordict" / "__init__.py").write_text(SHIM)
    _marker().write_text(COMMIT)


def gated_message(token):
    if not token:
        return ("Make 3D needs Meta's DINOv3 image encoder, which is free but gated: 1. on Hugging Face, open "
                f"{DINOV3_PAGE}, sign in and accept the licence; 2. in Protoart, save your Hugging Face access token "
                "(the token box in Tools → 🧊 Make 3D). Then press Make 3D again.")
    return (f"Your Hugging Face account doesn't have access to DINOv3 yet: open {DINOV3_PAGE} on Hugging Face and "
            "accept the licence with the same account as your saved token (approval can take a little while), then "
            "press Make 3D again.")


def download(emit):
    """The weights and both image encoders, each once (here in the app's process: its downloads go through curl when
    Python's own connections are blocked) -> their local folders."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    token = os.environ.get("HF_TOKEN")
    got = {}
    for key, repo, patterns, label, size in (
            ("dinov3", DINOV3, ["*.json", "*.safetensors"], "Meta's DINOv3 image encoder", "~1.7 GB"),
            ("vae", FLUX2_VAE, ["vae/*"], "the FLUX.2 image encoder", "~170 MB"),
            ("weights", WEIGHTS, WEIGHT_FILES, "Idol", "~3 GB")):
        local = _cached(repo, patterns)
        if not local:
            emit({"type": "status", "msg": f"Make 3D: downloading {label} ({size}, first time only)…"})
            try:
                local = snapshot_download(repo, allow_patterns=patterns, token=token or None)
            except (GatedRepoError, RepositoryNotFoundError):
                raise IdolError(gated_message(token))
            except Exception as e:
                if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                    raise IdolError(gated_message(token))
                raise IdolError(f"Couldn't download {label} ({e}). Check the internet connection and try again.")
        got[key] = str(Path(local).resolve())
    return got


def run(image_path, out_stem, opts, emit, cancel_event=None):
    """The worker turns a picture into out_stem.ply / _preview.png / _input.png -> its result dict."""
    import net
    if ram_gb() < MIN_RAM_GB:
        raise IdolError(f"Make 3D needs a Mac with 16 GB of memory or more (this one has {ram_gb():.0f} GB).")
    local = None
    if not all(models_downloaded().values()):
        if net.strict:
            raise IdolError(f"Make 3D needs a one-time download (~{DOWNLOAD_GB:.0f} GB), and Strict offline mode is "
                            "on. Turn it off in Settings → Privacy.")
        net.set_online(True)
        try:
            local = download(emit)
        finally:
            net.set_online(False)
    else:
        local = download(emit)  # all cached: just their folders
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", ATTN_BACKEND="sdpa",
               PYTORCH_ENABLE_MPS_FALLBACK="1", PROTOART_IDOL_SITE=str(site()), PROTOART_IDOL_CODE=str(code_dir()))
    args = dict(opts, image=str(image_path), out=str(out_stem), weights=local["weights"], dinov3=local["dinov3"],
                vae=local["vae"])
    p = subprocess.Popen([sys.executable, str(HERE / "idol_worker.py"), json.dumps(args)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=str(HERE), env=env)
    tail = []

    def read_err():
        for line in p.stderr:
            tail.append(line.rstrip())
            del tail[:-40]
            print("[idol]", line.rstrip(), file=sys.stderr, flush=True)

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
        raise IdolError((result or {}).get("msg") or "\n".join(tail[-6:]) or "Make 3D stopped")
    return result


def read_ply(path, max_points=150000):
    """A 3D Gaussian splat .ply (as Idol writes it) -> a lighter set of coloured points for the app's viewer:
    {"n", "pos" (float32 x,y,z), "col" (uint8 r,g,b,a), "size" (float32)}, each base64."""
    import base64

    import numpy as np
    with open(path, "rb") as f:
        names, count, fmt = [], 0, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("not a .ply file")
            s = line.decode("ascii", "replace").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element vertex"):
                count = int(s.split()[2])
            elif s.startswith("property"):
                bits = s.split()
                if bits[1] != "float":
                    raise ValueError("unexpected .ply layout")
                names.append(bits[2])
            elif s == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError("unexpected .ply format")
        data = np.frombuffer(f.read(count * 4 * len(names)), dtype="<f4").reshape(count, len(names))
    col = {n: i for i, n in enumerate(names)}
    if count > max_points:
        keep = np.random.default_rng(0).choice(count, max_points, replace=False)
        data = data[np.sort(keep)]
    xyz = data[:, [col["x"], col["y"], col["z"]]].astype(np.float32)
    c0 = 0.28209479177387814
    rgb = np.clip(0.5 + c0 * data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]], 0, 1)
    alpha = 1 / (1 + np.exp(-data[:, col["opacity"]]))
    size = np.exp(data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1).astype(np.float32)
    rgba = np.concatenate([rgb, alpha[:, None]], 1)
    rgba = (rgba * 255).round().astype(np.uint8)
    keep = alpha > 0.05
    xyz, rgba, size = xyz[keep], rgba[keep], size[keep]
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    return {"n": int(len(xyz)), "total": int(count), "pos": b64(xyz), "col": b64(rgba), "size": b64(size)}


def render_preview(xyz, rgb, alpha, scale, size=768, yaw_deg=-30, pitch_deg=12):
    """A picture of the splat (y-up points), turned a little to the side: points drawn back to front as small soft
    dots. Plain numpy, good enough for a Library thumbnail."""
    import numpy as np
    from PIL import Image
    v = xyz - (xyz.max(0) + xyz.min(0)) / 2
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    v = v @ ry.T @ rx.T
    span = max(1e-6, float((v[:, :2].max(0) - v[:, :2].min(0)).max()))
    k = 0.86 * size / span
    px = (size / 2 + v[:, 0] * k).astype(np.int64)
    py = (size / 2 - v[:, 1] * k).astype(np.int64)
    rad = np.clip(np.round(scale * k), 0, 2).astype(np.int64)
    order = np.argsort(v[:, 2])  # far (negative z, away from the viewer) first
    img = np.zeros((size, size, 3), np.float32) + np.array([46, 48, 52], np.float32) / 255
    for r in (2, 1, 0):  # bigger dots first, the fine ones on top
        sel = order[(rad[order] >= r) & (alpha[order] > 0.1)]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = px[sel] + dx, py[sel] + dy
                ok = (x >= 0) & (x < size) & (y >= 0) & (y < size)
                a = alpha[sel][ok][:, None]
                img[y[ok], x[ok]] = img[y[ok], x[ok]] * (1 - a) + rgb[sel][ok] * a
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
