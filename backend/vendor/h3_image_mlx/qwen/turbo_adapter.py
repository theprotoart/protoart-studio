"""Prepare the non-block portion of the pinned Comfy Turbo adapter on a private copy."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def header(stream):
    length = int.from_bytes(stream.read(8), "little")
    return json.loads(stream.read(length)), 8 + length


def raw(stream, offset, tensor):
    start, end = tensor["data_offsets"]
    stream.seek(offset + start)
    data = stream.read(end - start)
    if len(data) != end - start:
        raise ValueError("Turbo LoRA 权重不完整")
    return data


def array(stream, offset, tensor):
    if tensor["dtype"] != "BF16":
        raise ValueError("需要 BF16 Turbo LoRA")
    bits = np.frombuffer(raw(stream, offset, tensor), "<u2").astype("<u4") << 16
    return bits.view("<f4").reshape(tensor["shape"])


def pack_bfloat(x):
    bits = np.ascontiguousarray(x, dtype="<f4").view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()


def write_blocks(file, offset, tensors, keys, path):
    manifest = {"__metadata__": tensors["__metadata__"]}
    pos = 0
    for k in keys:
        tensor = tensors[k]
        start, end = tensor["data_offsets"]
        length = end - start
        manifest[k] = {**tensor, "data_offsets": [pos, pos + length]}
        pos += length
    data = json.dumps(manifest, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    with path.open("wb") as out:
        out.write(len(data).to_bytes(8, "little"))
        out.write(data)
        for k in keys:
            out.write(raw(file, offset, tensors[k]))


def prepare_turbo(root, base, adapter, strength):
    cache = root / "turbo-cache"
    cache.mkdir(exist_ok=True, parents=True)
    blocks = cache / "blocks.safetensors"
    current = cache / "current"
    if (
        (current / "scale.json").is_file()
        and blocks.is_file()
        and json.loads((current / "scale.json").read_text()) == strength
    ):
        return current / "transformer", blocks
    with tempfile.TemporaryDirectory(dir=cache) as folder, adapter.open("rb") as f:
        temp = Path(folder)
        tensors, offset = header(f)
        dynamic = [
            k for k in tensors if "transformer_blocks." in k and not k.endswith(".diff")
        ]
        rest = set(tensors) - set(dynamic) - {"__metadata__"}
        if len(dynamic) != 576 or len(rest) != 89:
            raise ValueError("Turbo LoRA 权重清单不匹配")
        if not blocks.exists():
            write_blocks(f, offset, tensors, dynamic, temp / "blocks")
            (temp / "blocks").replace(blocks)
        dest = temp / "transformer"
        dest.mkdir()
        shutil.copyfile(base / "config.json", dest / "config.json")
        seen = set()
        for src in sorted(base.glob("*.safetensors")):
            dst = dest / src.name
            subprocess.run(["/bin/cp", "-c", str(src.resolve()), str(dst)], check=True)
            with dst.open("r+b") as w:
                base_header, base_offset = header(w)
                for k in sorted(rest):
                    if not k.endswith((".alpha", ".diff")):
                        continue
                    name = (
                        k.removeprefix("diffusion_model.").rsplit(".", 1)[0] + ".weight"
                    )
                    if name not in base_header:
                        continue
                    tensor = base_header[name]
                    original = array(w, base_offset, tensor)
                    if k.endswith(".diff"):
                        delta = array(f, offset, tensors[k])
                        seen.add(k)
                    else:
                        down = k[:-6] + ".lora_down.weight"
                        up = k[:-6] + ".lora_up.weight"
                        a = array(f, offset, tensors[down])
                        b = array(f, offset, tensors[up])
                        delta = (b @ a) * (
                            float(array(f, offset, tensors[k])) / a.shape[0]
                        )
                        seen.update((k, up, down))
                    if delta.shape != original.shape:
                        raise ValueError("Turbo LoRA 修正尺寸不匹配")
                    w.seek(base_offset + tensor["data_offsets"][0])
                    w.write(pack_bfloat(original + delta * strength))
        if seen != rest:
            raise ValueError("Turbo LoRA 修正覆盖不完整")
        (temp / "scale.json").write_text(json.dumps(strength))
        # The cache is rebuildable. Publish only after both shards are ready.
        if current.exists():
            shutil.rmtree(current)
        temp.rename(current)
    return current / "transformer", blocks
