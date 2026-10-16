"""Merge LoRAs (Train tab → Merge LoRAs).

Two or more LoRAs made for the same model type become one new LoRA file. The originals are never changed.

  * "exact" (default): the LoRAs are stacked side by side (their ranks add up), each scaled by its weight, so the
    merged file does exactly what using them together at those weights does: just in one file.
  * "smaller": the combined change of each layer is squeezed to a chosen rank with a singular value decomposition
    (a close approximation, much smaller file).

Supported: standard LoRA layouts (kohya `lora_down/lora_up/alpha`, diffusers / PEFT `lora_A/lora_B`, old diffusers
`lora.down/lora.up`), linear and convolution layers. LyCORIS variants (LoHa, LoKr, DoRA, LoCon with a mid layer) are
refused with a clear message, since they can't be stacked this way. Pure torch + safetensors, runs on the CPU.
"""
import json
import re
from pathlib import Path

PAIRS = (  # (down suffix, up suffix, layout)
    (".lora_down.weight", ".lora_up.weight", "kohya"),
    (".lora_A.weight", ".lora_B.weight", "peft"),
    (".lora.down.weight", ".lora.up.weight", "diffusers-old"),
)
UNSUPPORTED = ("hada_", "lokr_", "dora_scale", "lora_mid", ".diff", "oft_", "boft_")


class MergeError(Exception):
    pass


def _layer_name(key):
    for d, u, fmt in PAIRS:
        if key.endswith(d):
            return key[: -len(d)], "down", fmt
        if key.endswith(u):
            return key[: -len(u)], "up", fmt
    return None, None, None


def read_lora(path):
    """-> (layers {base: {"down", "up", "alpha"}}, layout, metadata). Raises MergeError for files it can't merge."""
    from safetensors import safe_open
    p = Path(path)
    if p.suffix.lower() != ".safetensors":
        raise MergeError(f"{p.name}: only .safetensors LoRAs can be merged (DiffusionBee's .tdict files can't).")
    layers, layouts = {}, set()
    with safe_open(str(p), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        bad = [k for k in keys if any(u in k for u in UNSUPPORTED)]
        if bad:
            raise MergeError(f"{p.name} is a LyCORIS-type LoRA (LoHa / LoKr / DoRA / LoCon); those can't be merged here.")
        for k in keys:
            base, part, fmt = _layer_name(k)
            if base is None:
                if k.endswith(".alpha"):
                    layers.setdefault(k[: -len(".alpha")], {})["alpha"] = float(f.get_tensor(k).float().item())
                continue
            layouts.add(fmt)
            layers.setdefault(base, {})[part] = f.get_tensor(k)
    layers = {b: v for b, v in layers.items() if "down" in v and "up" in v}
    if not layers:
        raise MergeError(f"{p.name} doesn't look like a LoRA.")
    if len(layouts) > 1:
        raise MergeError(f"{p.name} mixes LoRA layouts; it can't be merged.")
    return layers, layouts.pop(), meta


def _scale(layer):
    r = layer["down"].shape[0]
    a = layer.get("alpha")
    return (a / r) if a else 1.0


def plan(items):
    """items: [{"path", "weight"}] -> checks only (no merging): {"layers", "rank_exact", "layout", "sizes_mb"}."""
    if len(items) < 2:
        raise MergeError("Pick at least two LoRAs to merge.")
    layouts, union, ranks = set(), set(), 0
    for it in items:
        layers, layout, _ = read_lora(it["path"])
        layouts.add(layout)
        union |= set(layers)
        ranks += max(v["down"].shape[0] for v in layers.values())
    if len(layouts) > 1:
        raise MergeError("These LoRAs are stored in different layouts (" + " and ".join(sorted(layouts)) +
                         "), so they can't be stacked into one file. Merge LoRAs from the same source or trainer.")
    return {"layers": len(union), "rank_exact": ranks, "layout": layouts.pop()}


def merge(items, out_path, mode="exact", rank=32, emit=None, cancelled=None):
    """items: [{"path", "weight", "name"}]. Writes out_path (fp16 .safetensors). -> dict about the result."""
    import torch
    from safetensors.torch import save_file
    if len(items) < 2:
        raise MergeError("Pick at least two LoRAs to merge.")
    loaded = []
    for it in items:
        layers, layout, meta = read_lora(it["path"])
        loaded.append((layers, layout, meta, float(it.get("weight", 1.0)), it.get("name") or Path(it["path"]).stem))
    layouts = {x[1] for x in loaded}
    if len(layouts) > 1:
        raise MergeError("These LoRAs are stored in different layouts (" + " and ".join(sorted(layouts)) +
                         "), so they can't be stacked into one file. Merge LoRAs from the same source or trainer.")
    layout = layouts.pop()
    d_suf, u_suf = {fmt: (d, u) for d, u, fmt in PAIRS}[layout]
    bases = sorted(set().union(*[set(x[0]) for x in loaded]))
    out, top_rank = {}, 0
    for n, base in enumerate(bases):
        if cancelled and cancelled():
            raise MergeError("Stopped.")
        parts = [(l[0][base], l[3]) for l in loaded if base in l[0]]
        downs, ups = [], []
        for layer, w in parts:
            down, up = layer["down"].float(), layer["up"].float()
            if down.shape[1:] != parts[0][0]["down"].shape[1:] or up.shape[0] != parts[0][0]["up"].shape[0]:
                raise MergeError(f"The LoRAs don't fit together at layer {base} (made for different models?).")
            downs.append(down)
            ups.append(up * (w * _scale(layer)))
        if mode == "smaller":
            d0, u0 = downs[0], ups[0]
            delta = sum(u.reshape(u.shape[0], -1) @ d.reshape(d.shape[0], -1) for d, u in zip(downs, ups))
            r = max(1, min(int(rank), min(delta.shape)))
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            U, S, Vh = U[:, :r], S[:r], Vh[:r]
            sq = S.sqrt()
            new_up = (U * sq).reshape(u0.shape[0], r, *u0.shape[2:])
            new_down = (sq[:, None] * Vh).reshape(r, *d0.shape[1:])
        else:
            dim_up = 1
            new_down = torch.cat(downs, 0)
            new_up = torch.cat(ups, dim_up)
            r = new_down.shape[0]
        top_rank = max(top_rank, r)
        out[base + d_suf] = new_down.to(torch.float16).contiguous()
        out[base + u_suf] = new_up.to(torch.float16).contiguous()
        if layout == "kohya":
            out[base + ".alpha"] = torch.tensor(float(r))  # scale alpha / rank = 1: the weights are already in
        if emit and (n % 40 == 0 or n == len(bases) - 1):
            emit({"type": "progress", "image": 1, "images": 1, "step": n + 1, "steps": len(bases)})
    # metadata: base model info from the first, trigger-word tags from all, and what went in
    meta0 = loaded[0][2]
    meta = {k: v for k, v in meta0.items() if k.startswith(("ss_base_model", "ss_sd_model", "ss_v2", "modelspec.architecture",
                                                           "ss_resolution", "ss_clip_skip"))}
    tags = {}
    for _, _, m, _, _ in loaded:
        try:
            for folder, counts in json.loads(m.get("ss_tag_frequency") or "{}").items():
                for t, c in counts.items():
                    tags[t] = tags.get(t, 0) + int(c)
        except (ValueError, TypeError, AttributeError):
            pass
    if tags:
        meta["ss_tag_frequency"] = json.dumps({"merged": tags})
    meta.update({
        "ss_network_dim": str(top_rank), "ss_network_alpha": str(top_rank),
        "protoart_merged_from": json.dumps([{"name": x[4], "weight": x[3]} for x in loaded]),
        "protoart_merge_mode": mode,
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    save_file(out, str(tmp), metadata=meta)
    tmp.replace(out_path)
    return {"path": str(out_path), "name": out_path.stem, "layers": len(bases), "rank": top_rank,
            "size_mb": round(out_path.stat().st_size / 1e6, 1), "mode": mode, "layout": layout}


def safe_name(name):
    return re.sub(r"[^\w\- .]+", "_", (name or "").strip()).strip(" .") or "merged_lora"
