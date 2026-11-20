"""Your own LoRAs on MOSS-Image (run inside the MOSS worker, with the moss-image-mlx package).

moss-image-mlx reads one adapter format: kohya "diffusion_model.<module>.lora_down/lora_up". Adapters made for MOSS
come in a few more, so this reads them all and attaches each one unmerged (the picture model's weights stay untouched):

  kohya LoRA    <module>.lora_down.weight / .lora_up.weight [/ .alpha]
  PEFT LoRA     <module>.lora_A.weight / .lora_B.weight [/ .alpha]
  LyCORIS LoKr  <module>.lokr_w1 (or _w1_a/_w1_b) and .lokr_w2 (or _w2_a/_w2_b) [/ .alpha]

<module> is a path in the picture model (layers.N.proj.q, layers.N.proj.ffn_up_gate…), after an optional
"diffusion_model." / "transformer." / "model." prefix. LoKr is applied as (W1 ⊗ W2)·x = W1 · X · W2ᵀ with x reshaped
to X, so the big Kronecker matrix is never built (that would be ~50 MB per layer)."""
import re
from pathlib import Path

PREFIXES = ("diffusion_model.", "transformer.", "model.diffusion_model.", "model.", "base_model.model.")
_SUFFIX = re.compile(r"\.(lora_down\.weight|lora_up\.weight|lora_A\.weight|lora_B\.weight|alpha|dora_scale|"
                     r"lokr_w1|lokr_w2|lokr_w1_a|lokr_w1_b|lokr_w2_a|lokr_w2_b|lokr_t2|hada_\w+)$")


class FormatError(ValueError):
    pass


def group_keys(keys):
    """{module path: {part: key}} for every adapter key (prefixes stripped)."""
    out = {}
    for k in keys:
        m = _SUFFIX.search(k)
        if not m:
            continue
        name = k[: m.start()]
        for p in PREFIXES:
            if name.startswith(p):
                name = name[len(p):]
                break
        out.setdefault(name, {})[m.group(1)] = k
    return out


def kind_of(parts):
    if "lora_down.weight" in parts and "lora_up.weight" in parts:
        return "lora"
    if "lora_A.weight" in parts and "lora_B.weight" in parts:
        return "peft"
    if ("lokr_w1" in parts or "lokr_w1_a" in parts) and ("lokr_w2" in parts or "lokr_w2_a" in parts):
        return "tucker" if "lokr_t2" in parts else "lokr"
    if any(p.startswith("hada_") for p in parts):
        return "loha"
    return None


def lokr_scale(parts, w1_rank, w2_rank, alpha):
    """LyCORIS: both factors full → scale 1; otherwise alpha / rank of the split factor."""
    rank = w2_rank or w1_rank
    if not rank:
        return 1.0
    return (alpha / rank) if alpha else 1.0


def apply(model, path, strength=1.0, dtype=None, group="user"):
    """Attach the adapter in `path` to `model` (moss-image-mlx's DiT). Returns how many layers it changed."""
    import mlx.core as mx
    import mlx.nn as nn
    from moss_image_mlx.lora import LoRALinear, _get, _resolve, _set

    dtype = dtype or mx.bfloat16

    class LoKrLinear(nn.Module):
        def __init__(self, base, w1, w2, scale):
            super().__init__()
            self.base, self.w1, self.w2 = base, w1, w2  # (a1, b1), (a2, b2): out = a1*a2, in = b1*b2
            self.scale = scale
            self.base_scale = scale
            self.group = group

        def __call__(self, x):
            y = self.base(x)
            if self.scale == 0:
                return y
            a1, b1 = self.w1.shape
            a2, b2 = self.w2.shape
            lead = x.shape[:-1]
            xs = x.astype(self.w2.dtype).reshape(*lead, b1, b2)
            t = self.w1 @ (xs @ self.w2.T)  # (…, a1, a2)
            return y + t.reshape(*lead, a1 * a2).astype(y.dtype) * self.scale

    weights = mx.load(str(path))
    groups = group_keys(weights.keys())
    if not groups:
        raise FormatError(f"{Path(path).name} isn't a LoRA this app can read for MOSS-Image")
    done = 0
    for name, parts in sorted(groups.items()):
        kind = kind_of(parts)
        if kind in ("loha", "tucker"):
            raise FormatError(f"{Path(path).name} is a {'LoHa' if kind == 'loha' else 'LoKr (Tucker)'} file, which "
                              "MOSS-Image can't use yet (LoRA and LoKr files work).")
        if kind is None:
            continue
        try:
            parent, key = _resolve(model, name)
            base = _get(parent, key)
        except (AttributeError, KeyError, IndexError, ValueError, TypeError):
            raise FormatError(f"{Path(path).name} doesn't match MOSS-Image's layers ({name}): it was probably made "
                              "for another model type.")
        take = lambda part: weights.pop(parts[part]).astype(dtype)  # noqa: E731
        alpha = weights.pop(parts["alpha"], None) if "alpha" in parts else None
        alpha = float(alpha.item()) if alpha is not None else None
        if kind in ("lora", "peft"):
            down = take("lora_down.weight" if kind == "lora" else "lora_A.weight")
            up = take("lora_up.weight" if kind == "lora" else "lora_B.weight")
            mx.eval(down, up)
            rank = down.shape[0]
            mod = LoRALinear(base, down, up, strength * (alpha if alpha else rank) / rank, group)
        else:
            w1, r1 = factor(take, parts, "lokr_w1")
            w2, r2 = factor(take, parts, "lokr_w2")
            mx.eval(w1, w2)
            out_f, in_f = w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]
            want_out, want_in = layer_shape(base)
            if want_out and (out_f, in_f) != (want_out, want_in):
                raise FormatError(f"{Path(path).name} doesn't fit MOSS-Image's {name} layer "
                                  f"({out_f}×{in_f}, needs {want_out}×{want_in}).")
            mod = LoKrLinear(base, w1, w2, strength * lokr_scale(parts, r1, r2, alpha))
        _set(parent, key, mod)
        done += 1
    del weights
    mx.clear_cache()
    if not done:
        raise FormatError(f"{Path(path).name} has no LoRA layers this app can read")
    return done


def factor(take, parts, name):
    """One LoKr factor and its rank: stored whole (rank 0), or split into _a @ _b (rank = rows of _b)."""
    if name in parts:
        return take(name), 0
    a, b = take(name + "_a"), take(name + "_b")
    return a @ b, int(b.shape[0])


def layer_shape(layer):
    """(out, in) of a linear layer (plain or quantized, or an adapter around one), or (None, None)."""
    while hasattr(layer, "base"):
        layer = layer.base
    w = getattr(layer, "weight", None)
    if w is None:
        return None, None
    out_f, in_f = w.shape[0], w.shape[-1]
    bits = getattr(layer, "bits", None)
    if bits:  # quantized: packed into 32-bit words
        in_f = in_f * 32 // bits
    return out_f, in_f
