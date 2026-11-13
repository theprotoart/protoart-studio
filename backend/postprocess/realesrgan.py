"""Real-ESRGAN upscalers (xinntao/Real-ESRGAN, BSD-3-Clause; weights from its GitHub releases, downloaded on first use):
  * RealESRGAN_x4plus            photos and general pictures (RRDBNet, 23 blocks)
  * RealESRGAN_x4plus_anime_6B   anime and drawings (RRDBNet, 6 blocks: smaller, sharper lines)
  * realesr-animevideov3         anime, very fast (SRVGGNetCompact, ~2.4 MB)
Our own small re-implementation of the two network shapes (the same layer names as Real-ESRGAN's, so its checkpoints
load unchanged); the tiling is the app's ESRGAN one.
"""
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .esrgan_model import UpscalerESRGAN

_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download/"
MODELS = {
    "realesrgan_x4plus": {"label": "Real-ESRGAN x4plus", "kind": "rrdb", "blocks": 23, "file": "RealESRGAN_x4plus.pth",
                          "url": _REL + "v0.1.0/RealESRGAN_x4plus.pth", "mb": 64,
                          "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"},
    "realesrgan_anime6b": {"label": "Real-ESRGAN x4plus Anime 6B", "kind": "rrdb", "blocks": 6,
                           "file": "RealESRGAN_x4plus_anime_6B.pth", "url": _REL + "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                           "mb": 18, "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"},
    "realesr_animevideov3": {"label": "Real-ESRGAN AnimeVideo v3", "kind": "vgg", "convs": 16,
                             "file": "realesr-animevideov3.pth", "url": _REL + "v0.2.5.0/realesr-animevideov3.pth",
                             "mb": 3, "sha256": "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d"},
}


def _lrelu(x):
    return F.leaky_relu(x, 0.2, inplace=True)


class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

    def forward(self, x):
        x1 = _lrelu(self.conv1(x))
        x2 = _lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = _lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = _lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN's 4× RRDBNet (layer names as in basicsr)."""

    def __init__(self, blocks=23, nf=64, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf, gc) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = _lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = _lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(_lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN's compact 4× network (AnimeVideo v3, general x4v3)."""

    def __init__(self, convs=16, nf=64, scale=4):
        super().__init__()
        self.scale = scale
        layers = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        for _ in range(convs):
            layers += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        layers.append(nn.Conv2d(nf, 3 * scale * scale, 3, 1, 1))
        self.body = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = F.pixel_shuffle(out, self.scale)
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def _dir():
    import paths
    d = paths.HOME / "weights" / "realesrgan"
    d.mkdir(parents=True, exist_ok=True)
    return d


def downloaded(mid):
    m = MODELS.get(mid)
    return bool(m) and (_dir() / m["file"]).exists()


def fetch(mid, emit=None):
    """The model's file on this Mac (downloaded once from Real-ESRGAN's GitHub releases, curl first)."""
    m = MODELS[mid]
    dest = _dir() / m["file"]
    if dest.exists():
        return dest
    import net
    if net.strict:
        raise RuntimeError(f"{m['label']} needs a one-time download (~{m['mb']} MB), and Strict offline mode is on.")
    if emit:
        emit({"type": "status", "msg": f"Downloading {m['label']} (~{m['mb']} MB, first time only)…"})
    tmp = dest.with_suffix(".part")
    ok = False
    if net.CURL and os.path.exists(net.CURL):
        r = subprocess.run([net.CURL, "-sS", "-L", "--fail", "--connect-timeout", "20", "--max-time", "900",
                            "-A", "ProtoartStudio", "-o", str(tmp), m["url"]], capture_output=True, text=True)
        ok = r.returncode == 0 and tmp.exists()
    if not ok:
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(m["url"], headers={"User-Agent": "ProtoartStudio"}), timeout=60) as r, \
                open(tmp, "wb") as f:
            f.write(r.read())
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != m["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"The {m['label']} download was damaged; try again.")
    tmp.replace(dest)
    return dest


class RealESRGAN(UpscalerESRGAN):
    """A Real-ESRGAN model with the app's tiled upscaling. Real-ESRGAN works in RGB (the older ESRGANs in BGR)."""

    def __init__(self, mid, device, dtype, emit=None):
        self.mid = mid
        super().__init__(fetch(mid, emit), device, dtype)

    def _load(self, path):
        m = MODELS[self.mid]
        state = torch.load(path, weights_only=True, map_location="cpu")
        state = state.get("params_ema") or state.get("params") or state
        net = RRDBNet(m["blocks"]) if m["kind"] == "rrdb" else SRVGGNetCompact(m["convs"])
        net.load_state_dict(state, strict=True)
        return net.eval()

    def _run(self, arr):
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255)
        x = x.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            y = self.model(x)
        y = y.squeeze(0).float().clamp_(0, 1).mul_(255).round_().byte().cpu().numpy()
        return y.transpose(1, 2, 0)


def upscale(mid, img, factor, device, dtype, emit=None, tile=256, overlap=16):
    """img (PIL) upscaled factor× (the model works at 4×; other factors are resized from that)."""
    up = RealESRGAN(mid, device, dtype, emit)
    if emit:
        emit({"type": "status", "msg": f"Upscaling with {MODELS[mid]['label']}…"})
    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA") else None
    out = up.upscale_with_tiling(img.convert("RGB"), tile=tile, overlap=overlap)
    w, h = round(img.width * factor), round(img.height * factor)
    if out.size != (w, h):
        out = out.resize((w, h), Image.LANCZOS)
    if alpha is not None:
        out.putalpha(alpha.resize((w, h), Image.LANCZOS))
    return out
