"""Old .ckpt checkpoints → .safetensors copies.

A .ckpt is a Python "pickle": opening one the ordinary way can run code hidden inside it, and newer PyTorch
refuses many old ones outright ("weights only load failed"), so some older models wouldn't load.

This reads a .ckpt with a restricted unpickler that only rebuilds tensors and plain containers. Anything else
in the file (training callbacks, optimiser objects, or anything malicious) becomes an inert placeholder and is
never run. The weights are then written as a new .safetensors file in Protoart's checkpoints folder.
The original .ckpt is only read: never changed, moved or deleted.
"""
import collections
import os
import pickle
import shutil
from pathlib import Path

import paths


class _Inert:
    """Stands in for any class the file names that isn't a tensor or a plain container. Never runs anything."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Inert()


def _allowed():
    import torch
    import torch._utils as tu
    ok = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("collections", "defaultdict"): collections.defaultdict,
        ("torch._utils", "_rebuild_tensor_v2"): tu._rebuild_tensor_v2,
        ("torch._utils", "_rebuild_parameter"): tu._rebuild_parameter,
        ("torch.nn.parameter", "Parameter"): torch.nn.Parameter,
        ("builtins", "set"): set,
        ("builtins", "frozenset"): frozenset,
        ("builtins", "slice"): slice,
        ("builtins", "complex"): complex,
    }
    for name in ("_rebuild_tensor_v3", "_rebuild_parameter_with_state", "_rebuild_device_tensor_from_numpy"):
        if hasattr(tu, name):
            ok[("torch._utils", name)] = getattr(tu, name)
    for st in ("FloatStorage", "HalfStorage", "BFloat16Storage", "DoubleStorage", "IntStorage", "LongStorage",
               "ByteStorage", "CharStorage", "ShortStorage", "BoolStorage"):
        if hasattr(torch, st):
            ok[("torch", st)] = getattr(torch, st)
    if hasattr(torch.storage, "UntypedStorage"):
        ok[("torch", "UntypedStorage")] = torch.storage.UntypedStorage
    for dt in ("float16", "float32", "float64", "bfloat16", "int64", "int32", "int16", "int8", "uint8", "bool"):
        ok[("torch", dt)] = getattr(torch, dt)
    return ok


class _SafeModule:
    """Passed to torch.load as pickle_module: its Unpickler only builds tensors and plain containers."""

    def __init__(self):
        allowed = _allowed()

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                f = allowed.get((module, name))
                if f is not None:
                    return f
                return _Inert

        self.Unpickler = Unpickler
        self.load = lambda f, **kw: Unpickler(f, **kw).load()  # every pickle read goes through the safe one
        self.__name__ = "protoart_safe_pickle"


def read_weights(path):
    """-> {name: tensor} from a .ckpt / .pt, without running anything inside the file."""
    import torch
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=_SafeModule())
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"{Path(path).name} doesn't contain model weights this app can read.")
    out = {k: v for k, v in obj.items() if isinstance(k, str) and torch.is_tensor(v)}
    if not out:
        raise ValueError(f"{Path(path).name} doesn't contain model weights this app can read.")
    return out


def find_ckpts(dirs):
    """Old .ckpt checkpoints (over 1 GB) in the given folders (sub-folders included), with a converted copy if any."""
    seen, rows = set(), []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            continue
        try:
            files = list(root.rglob("*.ckpt"))
        except OSError:
            continue
        for f in files:
            if f.name.startswith("._") or str(f) in seen:
                continue
            seen.add(str(f))
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size < 1e9:
                continue
            out = target_for(f)
            rows.append({"path": str(f), "name": f.name, "gb": round(size / 1e9, 1),
                         "converted": str(out) if out.exists() else ""})
    return sorted(rows, key=lambda r: r["name"].lower())


def target_for(src):
    return paths.MODELS_DIR / (Path(src).stem + ".safetensors")


def convert(src, half=True, emit=None, cancel_event=None):
    """Write a .safetensors copy of a .ckpt in Protoart's checkpoints folder. -> {"path", "gb", "tensors"}"""
    import torch
    from safetensors.torch import save_file
    emit = emit or (lambda m: None)
    src = Path(src)
    if src.suffix.lower() not in (".ckpt", ".pt", ".pth"):
        raise ValueError("Pick a .ckpt file.")
    if not src.is_file():
        raise ValueError(f"Can't find {src.name} (is the drive connected?).")
    out = target_for(src)
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    need = src.stat().st_size * (0.6 if half else 1.1)
    free = shutil.disk_usage(paths.MODELS_DIR).free
    if free < need:
        raise ValueError(f"Not enough free space: about {need / 1e9:.1f} GB is needed, {free / 1e9:.1f} GB is free.")
    emit({"type": "status", "msg": f"Reading {src.name} (safely: nothing inside it is run)…"})
    emit({"type": "progress", "image": 1, "images": 1, "step": 1, "steps": 4})
    sd = read_weights(src)
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError()
    emit({"type": "status", "msg": f"Preparing {len(sd)} weights…"})
    emit({"type": "progress", "image": 1, "images": 1, "step": 2, "steps": 4})
    clean = {}
    for k, v in sd.items():
        if k.startswith(("model_ema.", "optimizer")):  # training leftovers: not needed to make pictures
            continue
        v = v.detach()
        if half and v.dtype == torch.float32:
            v = v.half()
        clean[k] = v.contiguous().clone() if v.storage_offset() or not v.is_contiguous() else v.contiguous()
    del sd
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError()
    emit({"type": "status", "msg": f"Writing {out.name}…"})
    emit({"type": "progress", "image": 1, "images": 1, "step": 3, "steps": 4})
    tmp = out.with_suffix(".safetensors.part")
    try:
        _dedupe_shared(clean)
        save_file(clean, str(tmp), metadata={"format": "pt", "converted_from": src.name, "by": "Protoart Studio"})
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    emit({"type": "progress", "image": 1, "images": 1, "step": 4, "steps": 4})
    return {"path": str(out), "gb": round(out.stat().st_size / 1e9, 2), "tensors": len(clean), "source": str(src)}


def _dedupe_shared(sd):
    """safetensors refuses tensors that share memory: give each its own copy."""
    seen = {}
    for k, v in list(sd.items()):
        try:
            ptr = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape))
        except Exception:
            continue
        if ptr in seen:
            sd[k] = v.clone()
        else:
            seen[ptr] = k
