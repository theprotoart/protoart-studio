"""4x ESRGAN upscaler used by the Finegrain enhancer: written for Protoart Studio (Apache-2.0).

The network follows the published ESRGAN generator (Wang et al., "ESRGAN: Enhanced Super-Resolution
Generative Adversarial Networks", ECCV 2018 workshops; reference code https://github.com/xinntao/ESRGAN,
Apache-2.0). Its layers are laid out like the widely used "old architecture" checkpoints
(`model.0` ... `model.10`), so existing 4x checkpoints such as 4x-UltraSharp load unchanged.

Large pictures are upscaled in overlapping tiles that are blended with smooth weights, which keeps
memory low on small Macs and leaves no visible seams.
"""
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


def _conv(cin, cout):
    return nn.Conv2d(cin, cout, kernel_size=3, padding=1)


class _DenseBlock(nn.Module):
    """Five 3x3 convolutions, each seeing the block input plus every earlier output (residual-dense)."""

    def __init__(self, ch=64, growth=32):
        super().__init__()
        # Each conv sits in a Sequential so the parameter names match the checkpoints (convN.0.weight).
        for i in range(4):
            setattr(self, f"conv{i + 1}", nn.Sequential(_conv(ch + i * growth, growth),
                                                         nn.LeakyReLU(0.2, inplace=True)))
        self.conv5 = nn.Sequential(_conv(ch + 4 * growth, ch))

    def forward(self, x):
        feats = [x]
        for i in range(4):
            feats.append(getattr(self, f"conv{i + 1}")(torch.cat(feats, 1)))
        return x + 0.2 * self.conv5(torch.cat(feats, 1))


class _RRDB(nn.Module):
    """Three dense blocks with a scaled residual around them."""

    def __init__(self, ch):
        super().__init__()
        self.RDB1, self.RDB2, self.RDB3 = _DenseBlock(ch), _DenseBlock(ch), _DenseBlock(ch)

    def forward(self, x):
        return x + 0.2 * self.RDB3(self.RDB2(self.RDB1(x)))


class _Residual(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.sub = body

    def forward(self, x):
        return x + self.sub(x)


class _Nearest2x(nn.Module):
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")


class ESRGAN4x(nn.Module):
    """Generator: head conv -> N RRDBs (+ conv) as a residual trunk -> two nearest 2x upsamples with
    convs -> two final convs. Indices in `model` match the old-architecture checkpoints."""

    def __init__(self, in_ch=3, out_ch=3, ch=64, blocks=23):
        super().__init__()
        trunk = nn.Sequential(*[_RRDB(ch) for _ in range(blocks)], _conv(ch, ch))
        act = lambda: nn.LeakyReLU(0.2, inplace=True)  # noqa: E731
        self.model = nn.Sequential(
            _conv(in_ch, ch),                 # 0
            _Residual(trunk),                 # 1
            _Nearest2x(), _conv(ch, ch), act(),   # 2, 3, 4
            _Nearest2x(), _conv(ch, ch), act(),   # 5, 6, 7
            _conv(ch, ch), act(),             # 8, 9
            _conv(ch, out_ch),                # 10
        )

    def forward(self, x):
        return self.model(x)


def checkpoint_shape(state):
    """(in_ch, out_ch, ch, blocks) read from an old-architecture checkpoint; refuses other layouts."""
    if "model.0.weight" not in state or "model.10.weight" not in state:
        raise ValueError("Unsupported upscaler checkpoint (expected the 4x ESRGAN 'old architecture' layout).")
    head, tail = state["model.0.weight"], state["model.10.weight"]
    block_ids = {int(m.group(1)) for k in state if (m := re.match(r"model\.1\.sub\.(\d+)\.RDB1\.", k))}
    if not block_ids:
        raise ValueError("Unsupported upscaler checkpoint (no RRDB blocks found).")
    return head.shape[1], tail.shape[0], head.shape[0], max(block_ids) + 1


def _blend_weights(h, w, overlap):
    """2-D weights that rise linearly over `overlap` pixels at every edge (never zero, so the borders of
    the whole picture still count fully once normalised)."""
    def ramp(n):
        r = np.ones(n, dtype=np.float32)
        k = min(overlap, n // 2)
        if k > 0:
            edge = (np.arange(k, dtype=np.float32) + 1) / (k + 1)
            r[:k] = edge
            r[-k:] = edge[::-1]
        return r
    return ramp(h)[:, None] * ramp(w)[None, :]


def _starts(total, tile, overlap):
    """Tile start offsets covering [0, total) with at least `overlap` pixels shared between neighbours."""
    if total <= tile:
        return [0]
    step = tile - overlap
    out = list(range(0, total - tile, step))
    out.append(total - tile)
    return sorted(set(out))


class UpscalerESRGAN:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.model = self._load(self.model_path)
        self.to(device, dtype)

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.upscale_without_tiling(img)

    def to(self, device: torch.device, dtype: torch.dtype):
        self.device, self.dtype = device, dtype
        self.model.to(device=device, dtype=dtype)

    @staticmethod
    def _load(path):
        state = torch.load(path, weights_only=True, map_location="cpu")
        in_ch, out_ch, ch, blocks = checkpoint_shape(state)
        net = ESRGAN4x(in_ch, out_ch, ch, blocks)
        net.load_state_dict(state)
        return net.eval()

    def _run(self, arr):
        """uint8 RGB array (h, w, 3) -> uint8 RGB array (4h, 4w, 3). These checkpoints expect BGR."""
        x = torch.from_numpy(np.ascontiguousarray(arr[:, :, ::-1].transpose(2, 0, 1))).float().div_(255)
        x = x.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            y = self.model(x)
        y = y.squeeze(0).float().clamp_(0, 1).mul_(255).round_().byte().cpu().numpy()
        return y.transpose(1, 2, 0)[:, :, ::-1]

    def upscale_without_tiling(self, img: Image.Image) -> Image.Image:
        return Image.fromarray(np.ascontiguousarray(self._run(np.asarray(img.convert("RGB")))), "RGB")

    def upscale_with_tiling(self, img: Image.Image, tile=512, overlap=64) -> Image.Image:
        src = np.asarray(img.convert("RGB"))
        h, w = src.shape[:2]
        if h <= tile and w <= tile:
            return self.upscale_without_tiling(img)
        scale = None
        acc = wsum = None
        for y0 in _starts(h, tile, overlap):
            for x0 in _starts(w, tile, overlap):
                patch = src[y0:y0 + tile, x0:x0 + tile]
                out = self._run(patch).astype(np.float32)
                if scale is None:
                    scale = out.shape[0] // patch.shape[0]
                    acc = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
                    wsum = np.zeros((h * scale, w * scale, 1), dtype=np.float32)
                ph, pw = out.shape[:2]
                wt = _blend_weights(ph, pw, overlap * scale)[:, :, None]
                ys, xs = y0 * scale, x0 * scale
                acc[ys:ys + ph, xs:xs + pw] += out * wt
                wsum[ys:ys + ph, xs:xs + pw] += wt
        result = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).round().astype(np.uint8)
        return Image.fromarray(result, "RGB")
