"""LyCORIS LoKr adapters for the fast engine's models (mflux), which reads only plain LoRA files.

Flow-DPO for Z-Image Turbo (and many Civitai adapters) are LoKr: each layer's change is a Kronecker product
W1 ⊗ W2 (`lokr_w1`, `lokr_w2`, or their `_a`/`_b` halves), and mflux stops with "missing required LoRA matrices".
These files are taken out of mflux's LoRA list (`is_lokr`) and attached here after the model is built, unmerged:
each matching linear layer gets a small wrapper that adds (W1 ⊗ W2)·x = W1 · X · W2ᵀ (x reshaped to X), so the big
Kronecker matrix is never built. Module paths are the file's keys after a "diffusion_model." / "transformer." prefix
(layers.3.attention.to_q, layers.3.attention.to_out.0 …), found on the model's transformer by attribute / list index.
"""
import json
import struct
from pathlib import Path

from moss_lora import FormatError, factor, group_keys, kind_of, layer_shape, lokr_scale


def is_lokr(path):
    """True when a .safetensors file holds LoKr layers (read from its header only)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100_000_000:
                return False
            keys = json.loads(f.read(n)).keys()
    except (OSError, ValueError, struct.error):
        return False
    return any(".lokr_w1" in k or ".lokr_w2" in k for k in keys)


def _resolve(root, name):
    """(parent, key, layer) for a dotted path like layers.3.attention.to_out.0 (list items by number)."""
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() and isinstance(obj, (list, tuple)) else getattr(obj, p)
    last = parts[-1]
    layer = obj[int(last)] if last.isdigit() and isinstance(obj, (list, tuple)) else getattr(obj, last)
    return obj, last, layer


def _set(parent, key, value):
    if key.isdigit() and isinstance(parent, list):
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def apply(model, path, strength=1.0):
    """Attach the LoKr file at `path` to `model` (an mflux model: its transformer). Returns the layers changed."""
    import mlx.core as mx
    import mlx.nn as nn

    class LoKrLinear(nn.Module):
        def __init__(self, base, w1, w2, scale):
            super().__init__()
            self.base, self.w1, self.w2, self.scale = base, w1, w2, scale

        def __call__(self, x):
            y = self.base(x)
            a1, b1 = self.w1.shape
            a2, b2 = self.w2.shape
            lead = x.shape[:-1]
            xs = x.astype(self.w2.dtype).reshape(*lead, b1, b2)
            t = (self.w1 @ (xs @ self.w2.T)).reshape(*lead, a1 * a2)
            return y + t.astype(y.dtype) * self.scale

    root = getattr(model, "transformer", None) or model
    weights = mx.load(str(path))
    groups = group_keys(weights.keys())
    done, missing = 0, []
    for name, parts in sorted(groups.items()):
        kind = kind_of(parts)
        if kind != "lokr":
            continue
        try:
            parent, key, base = _resolve(root, name)
        except (AttributeError, IndexError, ValueError, TypeError):
            missing.append(name)
            continue
        take = lambda part: weights[parts[part]].astype(mx.bfloat16)  # noqa: E731
        w1, r1 = factor(take, parts, "lokr_w1")
        w2, r2 = factor(take, parts, "lokr_w2")
        out_f, in_f = w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]
        want = layer_shape(base)
        if want[0] and (out_f, in_f) != want:
            raise FormatError(f"{Path(path).name} doesn't fit this model's {name} layer ({out_f}×{in_f}, needs {want[0]}×{want[1]}).")
        alpha = weights.get(parts["alpha"]) if "alpha" in parts else None
        alpha = float(alpha.item()) if alpha is not None else None
        mx.eval(w1, w2)
        _set(parent, key, LoKrLinear(base, w1, w2, strength * lokr_scale(parts, r1, r2, alpha)))
        done += 1
    del weights
    if not done:
        raise FormatError(f"{Path(path).name}: none of its layers are in this model"
                          + (f" (e.g. {missing[0]})" if missing else "") + ": it was probably made for another model type.")
    return done
