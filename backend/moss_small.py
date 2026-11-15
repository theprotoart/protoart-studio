"""MOSS-Image's small copy for 8 GB Macs: the recipe, shared by the app (moss_engine.py) and the worker that makes it
(moss_worker.py, in MOSS-Image's own Python). No imports, so both sides can use it."""


def small_spec(group_size, skip, num_layers):
    """The small copy's recipe in moss-image-mlx's quantization-spec format (stored in its config.json, so loading
    rebuilds exactly these layers)."""
    over = {k: {"bits": 8} for k in ("proj.q", "proj.k", "proj.v", "proj.o", "text_embedder", "layers.0.",
                                     f"layers.{num_layers - 1}.")}
    return {"bits": 4, "group_size": int(group_size), "mode": "affine", "skip": list(skip), "overrides": over}


def bits_for(spec, path):
    """The bits a layer gets under a spec (the same rule as moss_image_mlx.quantize: later overrides win)."""
    bits = spec["bits"]
    for key, extra in (spec.get("overrides") or {}).items():
        if key in path:
            bits = extra.get("bits", bits)
    return bits
