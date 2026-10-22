"""Merge two checkpoints into a new one (like A1111's Checkpoint Merger).

  weighted sum:    A × (1 − m) + B × m
  add difference:  A + (B − C) × m     (e.g. add what an inpainting / style model learned on top of another base)

Works on .safetensors checkpoints of the same type (SD 1.5 with SD 1.5, SDXL with SDXL). Weights are read and
written one at a time, so even two SDXL models (7 GB each) merge on an 8 GB Mac. The result is a new .safetensors
file in Protoart's checkpoints folder; the models you merged aren't changed.
"""
import os
from pathlib import Path

import paths


def merge(a, b, multiplier=0.5, mode="sum", c=None, name=None, half=True, emit=None, cancel_event=None):
    import torch
    from safetensors import safe_open
    emit = emit or (lambda m: None)
    m = float(max(0.0, min(1.0, multiplier)))
    files = [a, b] + ([c] if mode == "add" else [])
    for f in files:
        if not f or not str(f).endswith(".safetensors") or not os.path.isfile(f):
            raise ValueError(f"Pick .safetensors checkpoint files ({Path(str(f or '')).name or 'missing'} isn't one).")
    if mode == "add" and not c:
        raise ValueError("Add difference needs a third model (the one B was made from).")
    out_name = (name or f"{Path(a).stem}+{Path(b).stem}-{int(m * 100)}").strip().replace("/", "-")[:120]
    out = paths.MODELS_DIR / (out_name + ".safetensors")
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    need = os.path.getsize(a) * (0.55 if half else 1.05)
    if shutil.disk_usage(paths.MODELS_DIR).free < need:
        raise ValueError(f"Not enough free space for the merged model (~{need / 1e9:.1f} GB).")
    fa, fb = safe_open(a, "pt"), safe_open(b, "pt")
    fc = safe_open(c, "pt") if mode == "add" else None
    keys = [k for k in fa.keys()]
    shared = set(fb.keys())
    model_keys = [k for k in keys if k.startswith(("model.diffusion_model.", "first_stage_model.", "cond_stage_model.", "conditioner."))]
    if model_keys and len([k for k in model_keys if k in shared]) < 0.9 * len(model_keys):
        raise ValueError("These two models aren't the same type (SD 1.5 merges only with SD 1.5, SDXL with SDXL).")
    total = len(keys)
    fc_keys = set(fc.keys()) if fc is not None else set()

    def merged(k):
        ta = fa.get_tensor(k)
        if k in shared and not k.startswith("first_stage_model."):  # the VAE is kept from model A
            tb = fb.get_tensor(k)
            if tb.shape == ta.shape and ta.is_floating_point():
                if mode == "add":
                    if k in fc_keys:
                        tc = fc.get_tensor(k)
                        if tc.shape == ta.shape:
                            ta = ta.float() + (tb.float() - tc.float()) * m
                else:
                    ta = ta.float() * (1 - m) + tb.float() * m
        if ta.is_floating_point():
            ta = ta.half() if half else ta.to(torch.float32 if ta.dtype == torch.float64 else ta.dtype)
        return ta.contiguous()

    # The header needs every weight's type and shape before the data: they're known from model A.
    specs = []
    with safe_open(a, "pt") as probe:
        for k in keys:
            sl = probe.get_slice(k)
            dt = sl.get_dtype()
            if half and dt in ("F32", "F64", "BF16", "F16"):
                dt = "F16"
            elif dt == "F64":
                dt = "F32"
            specs.append((k, dt, list(sl.get_shape())))
    emit({"type": "status", "msg": "Merging…"})
    tmp = out.with_suffix(".safetensors.part")
    try:
        _stream_write(tmp, specs, merged, {"format": "pt", "merged_by": "Protoart Studio", "mode": mode,
                                           "multiplier": str(m), "a": Path(a).name, "b": Path(b).name},
                      emit, cancel_event)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    emit({"type": "progress", "image": 1, "images": 1, "step": total, "steps": total})
    return {"path": str(out), "gb": round(out.stat().st_size / 1e9, 2), "name": out.stem}


_SIZES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _stream_write(path, specs, get, metadata, emit, cancel_event):
    """Write a .safetensors file one tensor at a time (only one weight is in memory at once)."""
    import json
    import struct
    import torch
    header, offset = {"__metadata__": metadata}, 0
    for k, dt, shape in specs:
        n = _SIZES[dt]
        for d in shape:
            n *= d
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    want = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for i, (k, dt, shape) in enumerate(specs):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError()
            if i % 50 == 0:
                emit({"type": "progress", "image": 1, "images": 1, "step": i, "steps": len(specs)})
                emit({"type": "status", "msg": f"Merging… {int(100 * i / max(1, len(specs)))}%"})
            t = get(k)
            if dt in want and t.dtype != want[dt]:
                t = t.to(want[dt])
            if t.dtype == torch.bfloat16:
                data = t.view(torch.int16).numpy().tobytes()
            else:
                data = t.numpy().tobytes()
            f.write(data)
