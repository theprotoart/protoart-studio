"""LoKr files (Flow-DPO for Z-Image) are recognised and their layers found on an mflux model."""
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lokr_mlx  # noqa: E402


def _file(keys):
    head = json.dumps({k: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    f = Path(tempfile.mkdtemp()) / "x.safetensors"
    f.write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0")
    return f


def test_is_lokr():
    assert lokr_mlx.is_lokr(_file(["diffusion_model.layers.0.attention.to_q.lokr_w1", "diffusion_model.layers.0.attention.to_q.lokr_w2"]))
    assert not lokr_mlx.is_lokr(_file(["diffusion_model.layers.0.attention.to_q.lora_down.weight"]))
    assert not lokr_mlx.is_lokr(Path(tempfile.mkdtemp()) / "missing.safetensors")


def test_resolve_paths_like_mflux_z_image():
    class Lin:
        pass

    class Attn:
        def __init__(self):
            self.to_q, self.to_out = Lin(), [Lin()]

    class Block:
        def __init__(self):
            self.attention, self.adaLN_modulation = Attn(), [Lin()]

    class T:
        layers = [Block(), Block()]

    root = T()
    parent, key, layer = lokr_mlx._resolve(root, "layers.1.attention.to_out.0")
    assert layer is root.layers[1].attention.to_out[0] and key == "0"
    new = object()
    lokr_mlx._set(parent, key, new)
    assert root.layers[1].attention.to_out[0] is new
    _, key, layer = lokr_mlx._resolve(root, "layers.0.adaLN_modulation.0")
    assert layer is root.layers[0].adaLN_modulation[0]
