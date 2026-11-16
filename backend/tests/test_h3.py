"""MiniMax-H3-Image: the 4-bit copy's file layout, the licence gate, edit prompts and the catalog entry."""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_engine  # noqa: E402
import h3_worker  # noqa: E402
import paths  # noqa: E402


def _fake_int8(path):
    rng = np.random.default_rng(0)
    t = {"blocks.0.attn.qkv_proj.weight": rng.integers(-127, 127, (128, 512), dtype=np.int8),
         "blocks.0.attn.qkv_proj.weight_scale": rng.random(128).astype(np.float32),
         "blocks.0.attn.qkv_proj.comfy_quant": np.frombuffer(b'{"format":"int8_tensorwise"}', np.uint8),
         "blocks.0.norm.weight": rng.random(512).astype(np.float32),
         "model.embed_tokens.weight": rng.integers(-127, 127, (10, 512), dtype=np.int8),
         "model.embed_tokens.weight_scale": rng.random(10).astype(np.float32),
         "model.embed_tokens.comfy_quant": np.frombuffer(b"{}", np.uint8)}
    kinds = {np.int8: "I8", np.float32: "F32", np.uint8: "U8"}
    header, off, blobs = {"__metadata__": {"config": '{"x": 1}'}}, 0, []
    for k, v in t.items():
        header[k] = {"dtype": kinds[v.dtype.type], "shape": list(v.shape), "data_offsets": [off, off + v.nbytes]}
        off += v.nbytes
        blobs.append(v.tobytes())
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return t


def test_four_bit_layout():
    d = Path(tempfile.mkdtemp())
    t = _fake_int8(d / "in.safetensors")

    def fake(w8, scale):
        n, k = w8.shape
        return np.full((n, k // 8), 7, np.uint32), np.ones((n, k // 64), np.uint16), np.full((n, k // 64), 2, np.uint16)

    n, size = h3_worker.convert(d / "in.safetensors", d / "out.safetensors", fake)
    assert n == 1  # the projection; the embedding table stays as it was
    h, start = h3_worker.read_header(d / "out.safetensors")
    assert h["__metadata__"]["config"] == '{"x": 1}' and h["__metadata__"]["bits"] == "4"  # the original's kept
    data = (d / "out.safetensors").read_bytes()
    assert len(data) == start + size

    def get(k):
        a, b = h[k]["data_offsets"]
        return np.frombuffer(data[start + a:start + b], h3_worker.NP[h[k]["dtype"]]).reshape(h[k]["shape"])

    assert h["blocks.0.attn.qkv_proj.weight"]["dtype"] == "U32" and get("blocks.0.attn.qkv_proj.weight").shape == (128, 64)
    assert get("blocks.0.attn.qkv_proj.scales").shape == (128, 8) and "blocks.0.attn.qkv_proj.weight_scale" not in h
    assert (get("blocks.0.norm.weight") == t["blocks.0.norm.weight"]).all()
    assert (get("model.embed_tokens.weight") == t["model.embed_tokens.weight"]).all()
    assert not (d / "out.safetensors.part").exists()


def test_licence_gate():
    old = paths.HOME
    paths.HOME = Path(tempfile.mkdtemp())
    try:
        assert not h3_engine.licence_accepted()
        try:
            h3_engine.download(lambda ev: None)
            raise AssertionError("downloaded without the licence")
        except h3_engine.H3Error as e:
            assert "Accept" in str(e)
        h3_engine.accept_licence(True)
        assert h3_engine.licence_accepted()
        h3_engine.accept_licence(False)
        assert not h3_engine.licence_accepted()
    finally:
        paths.HOME = old


def test_edit_prompt():
    assert h3_engine.edit_prompt("make it night", 1) == "Edit <Picture 1>. make it night"
    assert h3_engine.edit_prompt("put the cat in the scene", 2).startswith("Edit <Picture 1> using <Picture 2>.")
    assert h3_engine.edit_prompt("Replace <Picture 1>'s sky", 1) == "Replace <Picture 1>'s sky"  # already names it
    assert h3_engine.edit_prompt("a cat", 0) == "a cat"


def test_catalog_and_manifest():
    import engine
    m = engine.all_models()["h3_image"]
    assert m["family"] == "h3" and m["h3"] and m["min_ram_gb"] == 8 and m["defaults"]["steps"] == 6
    assert len(h3_engine.FILES) == 5 and h3_engine.FILES[0]["path"].startswith("diffusion_models/")
    assert all(f["huggingface"].startswith("https://huggingface.co/") and f["modelscope"] for f in h3_engine.FILES)
    assert all(w % 32 == 0 and h % 32 == 0 for w, h in h3_engine.SIZES)


def test_memory_watchdog():
    base = (2.0, 6.0, 8.0)
    assert not h3_engine.starving(base, (1.5, 7.0, 8.0))  # some swapping is normal
    assert not h3_engine.starving(base, (2.5, 9.5, 8.0))  # lots of swap but RAM still free
    assert h3_engine.starving(base, (0.2, 9.5, 8.0))  # out of RAM and swapping hard: stop it
    assert not h3_engine.starving(base, None)
