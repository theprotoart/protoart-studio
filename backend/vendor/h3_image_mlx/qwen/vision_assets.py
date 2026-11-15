"""Derive the vision-only shard without allocating model tensors."""

import json
import struct


class Cancelled(RuntimeError):
    pass


def extract_vision(source, target, cancel=None):
    """Same tensor selection as prepare_vision.py, streaming without Torch/GPU."""
    with source.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        tensors = {
            k: v.copy() for k, v in header.items() if k.startswith("model.visual.")
        }
        if len(tensors) != 529:
            raise ValueError("Turbo vision tensor inventory mismatch")
        offset = 0
        for value in tensors.values():
            a, b = value["data_offsets"]
            value["data_offsets"] = [offset, offset + b - a]
            offset += b - a
        encoded = json.dumps(tensors, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".partial")
        with temp.open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for key in tensors:
                a, b = header[key]["data_offsets"]
                f.seek(8 + size + a)
                remaining = b - a
                while remaining:
                    if cancel and cancel.is_set():
                        raise Cancelled()
                    chunk = f.read(min(8 * 1024**2, remaining))
                    if not chunk:
                        raise ValueError("truncated vision source")
                    out.write(chunk)
                    remaining -= len(chunk)
        temp.replace(target)
