"""MiniMax-H3-Image's MLX side, run with the fast engine's Python (mlx_engine.PY) by h3_engine.py, one job per process
so each stage hands its memory back to macOS when it ends.

  python h3_worker.py convert <int8 file> <4-bit file>      the 4-bit copy (see convert())
  python h3_worker.py stage text|refs|dit|vae <work folder> one stage of one picture (request.json in the folder)

Progress and results are JSON lines on stdout. Inference is h3-image-mlx (vendor/h3_image_mlx, Apache-2.0, unmodified);
the only addition here is reading the 4-bit projections (patch())."""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
BITS, GROUP = 4, 64
NP = {"I8": np.int8, "U8": np.uint8, "F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I64": np.int64,
      "U32": np.uint32}


def out(**kw):
    print(json.dumps(kw), flush=True)


# ------------------------------------------------------------------------------------------------- the 4-bit copy
def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def quant_prefixes(h):
    """INT8 ConvRot projections (the big matrices): these become 4-bit. Embeddings and everything else are copied."""
    found = set()
    for k in h:
        if not k.endswith(".comfy_quant"):
            continue
        p = k[: -len(".comfy_quant")]
        w = h.get(p + ".weight")
        if w and w["dtype"] == "I8" and p + ".weight_scale" in h and "embed" not in p and len(w["shape"]) == 2 \
                and w["shape"][1] % 256 == 0:
            found.add(p)
    return found


def plan(h):
    q = quant_prefixes(h)
    items = []
    for k, spec in sorted(((k, v) for k, v in h.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0]):
        p = k.rsplit(".", 1)[0]
        if p in q:
            if k == p + ".weight":
                n, kd = spec["shape"]
                items += [(p + ".weight", "U32", [n, kd * BITS // 32], ("q", p)),
                          (p + ".scales", "BF16", [n, kd // GROUP], ("s", p)),
                          (p + ".biases", "BF16", [n, kd // GROUP], ("b", p))]
            continue  # .weight_scale and .comfy_quant are folded into the 4-bit numbers
        items.append((k, spec["dtype"], spec["shape"], ("copy", k)))
    return items, q


def nbytes(dtype, shape):
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(NP[dtype]).itemsize


def read(f, start, spec):
    a, b = spec["data_offsets"]
    f.seek(start + a)
    return np.frombuffer(f.read(b - a), NP[spec["dtype"]]).reshape(spec["shape"])


def quantize(w8, scale):
    """INT8 rows (already rotated by ConvRot's grouped Hadamard) x their scale -> MLX affine 4-bit, still rotated: at
    run time the input is rotated the same way (patch()). The rotation spreads outliers, which also suits 4-bit."""
    import mlx.core as mx
    s = scale.astype(np.float32).reshape(-1)
    if s.size not in (1, w8.shape[0]):
        raise ValueError(f"unexpected scale shape {scale.shape} for {w8.shape}")
    w = mx.array(w8).astype(mx.float32) * mx.array(s if s.size == 1 else s[:, None])
    wq, sc, bi = mx.quantize(w.astype(mx.bfloat16), group_size=GROUP, bits=BITS)
    mx.eval(wq, sc, bi)
    res = np.array(wq), np.array(sc.view(mx.uint16)), np.array(bi.view(mx.uint16))
    del w, wq, sc, bi
    mx.clear_cache()
    return res


def convert(src, dst, quantizer=quantize, report=lambda done, total: None):
    """Write the 4-bit copy one tensor at a time (memory stays low). Keeps the original's metadata (the picture model's
    sampling config lives there). -> (projections made 4-bit, bytes written)"""
    src, dst = Path(src), Path(dst)
    h, start = read_header(src)
    items, q = plan(h)
    meta = dict(h.get("__metadata__") or {})
    meta.update({"protoart": "h3-image-4bit-convrot", "bits": str(BITS), "group_size": str(GROUP), "source": src.name})
    header, off = {"__metadata__": meta}, 0
    for name, dt, shape, _ in items:
        n = nbytes(dt, shape)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    part = dst.with_name(dst.name + ".part")
    done, pending = 0, {}
    with open(src, "rb") as f, open(part, "wb") as o:
        o.write(struct.pack("<Q", len(raw)) + raw)
        for i, (name, dt, shape, (kind, key)) in enumerate(items):
            if kind == "copy":
                arr = read(f, start, h[key])
            elif kind == "q":
                pending = dict(zip("qsb", quantizer(read(f, start, h[key + ".weight"]),
                                                    read(f, start, h[key + ".weight_scale"]))))
                arr = pending["q"]
            else:
                arr = pending.pop(kind)
            arr = np.ascontiguousarray(arr)
            if arr.nbytes != nbytes(dt, shape) or list(arr.shape) != list(shape):
                raise ValueError(f"{name}: got {arr.shape}/{arr.nbytes}, planned {shape}")
            o.write(arr.tobytes())
            done += arr.nbytes
            if i % 20 == 0:
                report(done, off)
    part.replace(dst)
    report(off, off)
    return len(q), off


# --------------------------------------------------------------------------------------------------- inference
def patch():
    """Teach h3-image-mlx's weight reader the 4-bit projections made by convert() (its own files stay unchanged)."""
    import mlx.core as mx
    from h3_image_mlx import core
    core.DTYPES.setdefault("U32", "<u4")
    plain = core.Weights.linear

    def linear(self, prefix, x):
        if prefix + ".scales" not in self.header:
            return plain(self, prefix, x)
        w, s, b = self[prefix + ".weight"], self[prefix + ".scales"], self[prefix + ".biases"]
        if s.dtype != mx.bfloat16:
            s, b = s.view(mx.bfloat16), b.view(mx.bfloat16)
        xr = (x.astype(mx.float32).reshape(-1, 256) @ core.H256).reshape(x.shape).astype(x.dtype)
        y = mx.quantized_matmul(xr, w, s, b, transpose=True, group_size=GROUP, bits=BITS)
        if self.lora is not None:
            extra = self.lora.delta(prefix, x)
            if extra is not None:
                y = y + extra.astype(y.dtype)
        if prefix + ".bias" in self.header:
            y = y + self[prefix + ".bias"].astype(y.dtype)
        return y

    core.Weights.linear = linear
    return core


def size_refs(sides):
    """Reference pictures go in at the size asked for (h3-image-mlx enlarges each to at least 1024 on its short side:
    six pictures then cost ~6,000 tokens a step, 44 minutes on an 8 GB M1). sides: {path: short side}."""
    if not sides:
        return
    from PIL import Image, ImageOps
    from h3_image_mlx import ref_vae, vision
    plain = vision.reference_image

    def reference_image(path, max_short_edge):
        cap = sides.get(str(path))
        if not cap:
            return plain(path, max_short_edge)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            scale = min(int(cap), max_short_edge) / min(im.size)
            size = tuple(max(32, round(v * scale / 32) * 32) for v in im.size)
            return np.asarray(im.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255

    vision.reference_image = reference_image
    ref_vae.reference_image = reference_image


def stage(name, work):
    import mlx.core as mx
    core = patch()
    core.backend.set("portable")
    mx.set_cache_limit(96 << 20)
    work = Path(work)
    req = json.loads((work / "request.json").read_text())
    p = {k: Path(v) for k, v in req["paths"].items()}
    refs = [Path(r) for r in req.get("refs") or []]
    size_refs({str(r): s for r, s in zip(refs, req.get("ref_sides") or []) if s})
    t = time.time()
    if name == "text":
        from h3_image_mlx.text import encode
        x, tags, _ = encode(p["text"], p["tokenizer"], req["prompt"], refs)
        mx.save_safetensors(str(work / "embedding.safetensors"), {"embedding": x, "tags": tags})
    elif name == "refs":
        from h3_image_mlx.ref_vae import encode_reference
        w = core.Weights(p["vae"], True)
        stats = json.loads(p["vae_config"].read_text())
        zs = {}
        for i, r in enumerate(refs):
            zs[str(i)] = encode_reference(w, r, stats)
            mx.eval(zs[str(i)])
            out(type="progress", stage="refs", step=i + 1, steps=len(refs))
        mx.save_safetensors(str(work / "references.safetensors"), zs)
    elif name == "dit":
        from h3_image_mlx.dit import denoise
        e = mx.load(str(work / "embedding.safetensors"))
        zs = []
        if refs:
            images = mx.load(str(work / "references.safetensors"))
            zs = [images[str(i)] for i in range(len(refs))]
        z, _ = denoise(p["dit"], e["embedding"], req["width"], req["height"], req["seed"], False, zs, e["tags"],
                       progress=lambda step, secs: out(type="progress", stage="dit", step=step, steps=6,
                                                       seconds=round(secs, 1)))
        mx.save_safetensors(str(work / "latent.safetensors"), {"latent": z})
    elif name == "vae":
        from h3_image_mlx.vae import decode
        stats = json.loads(p["vae_config"].read_text())
        decode(p["vae"], mx.load(str(work / "latent.safetensors"))["latent"], stats).save(work / "image.png")
    else:
        raise SystemExit(f"unknown stage {name}")
    mx.synchronize()
    out(type="done", stage=name, seconds=round(time.time() - t, 1), peak_gb=round(mx.get_peak_memory() / 2**30, 2))


if __name__ == "__main__":
    if sys.argv[1] == "convert":
        n, size = convert(sys.argv[2], sys.argv[3], report=lambda d, tot: out(type="progress", done=d, total=tot))
        out(type="done", projections=n, bytes=size)
    elif sys.argv[1] == "stage":
        stage(sys.argv[2], sys.argv[3])
