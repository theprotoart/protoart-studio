"""Private stage entry point. Each child releases its weights on exit."""
import json
import os
import re
import sys
from pathlib import Path
from .models import model_paths, manifest_for


def select_backend(requested, device, precision="int8"):
    manifest_for(precision)
    if precision == "bf16":
        if requested == "w8a8":
            raise ValueError("W8A8 requires INT8 weights; use --backend auto or portable with BF16")
        return "portable"
    native = bool(re.search(r"\bApple M[56]\b", device))
    if requested == "auto":
        return "w8a8" if native else "portable"
    if requested == "w8a8" and not native:
        raise ValueError(f"The W8A8 TensorOps kernel requires an M5/M6 GPU; found {device}")
    return requested


def main(stage, work):
    import mlx.core as mx
    from . import core
    request = json.loads((work / "request.json").read_text())
    info = mx.device_info()
    precision = request.get("precision", "int8")
    dit, text, vae, tokenizer, vae_config = model_paths(precision)
    backend = select_backend(request["backend"], info["device_name"], precision)
    core.backend.set(backend)
    mx.set_cache_limit(96 << 20)
    models = Path(request["models"])
    refs = request["references"]
    embedding, latents, result = (str(work / (key + ".safetensors")) for key in ("embedding", "references", "latent"))
    mode = request.get("dit_mode")
    auto_resident = mode == "auto" and backend == "w8a8" and info["memory_size"] >= 32 * 1024**3
    # The exact DiT sequence length becomes available after encoding.
    m5 = backend == "w8a8" and bool(re.search(r"\bApple M5\b", info["device_name"]))
    likely_resident = auto_resident and not refs and request["width"] * request["height"] // 1024 < 8192
    core.m5_resident.set(m5 and (mode == "resident" or likely_resident))
    if stage == "text":
        from .text import encode
        # H3_TEXT / H3_TOKENIZER / H3_VAE / H3_DIT: use another checkpoint (e.g. P-H3-Image) instead of the manifest's.
        x, tags, _ = encode(os.environ.get("H3_TEXT") or models / text, os.environ.get("H3_TOKENIZER") or models / tokenizer, request["prompt"], refs)
        mx.save_safetensors(embedding, {"embedding": x, "tags": tags})
    elif stage == "references":
        from .ref_vae import encode_reference
        weights = core.Weights(models / vae, True)
        stats = json.loads((models / vae_config).read_text())
        zs = {}
        for i, ref in enumerate(refs):
            zs[str(i)] = encode_reference(weights, ref, stats)
            mx.eval(zs[str(i)])
        mx.save_safetensors(latents, zs)
    elif stage == "dit":
        from .dit import denoise
        inputs = mx.load(embedding)
        images = mx.load(latents) if refs else {}
        zs = [images[str(i)] for i in range(len(refs))]
        width, height = request["width"], request["height"]
        rows = inputs["embedding"].shape[0] + width * height // 1024 + sum(z.shape[1] * z.shape[2] // 4 for z in zs)
        resident = mode == "resident" or (auto_resident and rows <= 8192)
        core.m5_resident.set(m5 and resident)
        # H3_DIT=<file>: use another DiT checkpoint with the same layout (e.g. one with a LoRA merged in).
        z, _ = denoise(os.environ.get("H3_DIT") or models / dit, inputs["embedding"], width, height, request["seed"], resident, zs, inputs["tags"], precision=precision)
        mx.save_safetensors(result, {"latent": z})
    elif stage == "vae":
        from .vae import decode
        stats = json.loads((models / vae_config).read_text())
        decode(os.environ.get("H3_VAE") or models / vae, mx.load(result)["latent"], stats).save(work / "image.png")
    else:
        raise ValueError(f"Unknown stage: {stage}")
    mx.synchronize()


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
