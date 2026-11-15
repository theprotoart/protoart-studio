import math

import numpy as np
from .core import Weights, mx, norm, rope, swiglu
from PIL import Image


def tile_positions(size):
    if size <= 16:
        return [0], size
    count = math.ceil((size - 4) / 12)
    extra = 16 * count - 4 * (count - 1) - size
    gaps = [
        12 - extra // (count - 1) - (i < extra % (count - 1)) for i in range(count - 1)
    ]
    return [0, *np.cumsum(gaps).tolist()], 16


# P-H3-Image decodes one frame at a fixed rotary time coordinate; the H3 Image decoder has no time axis.
PH3_FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0


def decode_tiles(w, z):
    batch, h, width, channels = z.shape
    z = z.reshape(batch, -1, channels)
    kind = z.dtype
    dim = w.header["decoder.x_embedder.weight"]["shape"][0]
    layers = sum(key.endswith(".attn.to_qkv.weight") for key in w.header)
    x = z @ w["post_quant_conv.weight"].reshape(channels, channels).astype(kind).T + w["post_quant_conv.bias"].astype(kind)
    x = w.linear("decoder.x_embedder", x)
    x = mx.concatenate(
        [
            x,
            mx.broadcast_to(w["decoder.register_tokens"].astype(kind), (batch, 4, dim)),
            mx.zeros_like(x[:, :1]),
        ],
        axis=1,
    )
    coords = np.meshgrid(
        (np.arange(h) + 0.5) * 2 / h - 1,
        (np.arange(width) + 0.5) * 2 / width - 1,
        indexing="ij",
    )
    positions = np.stack(coords, axis=-1).reshape(-1, 2)
    if "latents_mean" in w.header:  # P-H3-Image: rotary axes (t, h, w) on 48 of 64 head channels
        positions = np.concatenate([np.full((len(positions), 1), PH3_FRAME_POSITION), positions], axis=1)
    positions = np.concatenate([positions, np.zeros((5, positions.shape[1]))])
    ang = (2 * np.pi * positions[:, :, None] * 100 ** (-np.arange(8) / 8)).reshape(
        len(positions), -1
    )
    angles = mx.array(ang.astype(np.float32))
    for i in range(layers):
        b = f"decoder.transformer_blocks.{i}"
        qkv = w.linear(b + ".attn.to_qkv", norm(x, w[b + ".norm1.weight"])).reshape(
            batch, x.shape[1], dim // 64, 192
        )
        q, k, v = mx.split(qkv.transpose(0, 2, 1, 3), 3, axis=-1)
        q, k = rope(norm(q), angles), rope(norm(k), angles)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
        x = (
            x
            + w.linear(b + ".attn.to_out", out.transpose(0, 2, 1, 3).reshape(x.shape))
            * w[b + ".scale1"].astype(kind)
        )
        out = swiglu(w.linear(b + ".ff.w1", norm(x, w[b + ".norm2.weight"])))
        x = x + w.linear(b + ".ff.w2", out) * w[b + ".scale2"].astype(kind)
        mx.eval(x)
    x = mx.fast.layer_norm(
        x[:, : h * width],
        w["decoder.norm_out.weight"].astype(kind),
        w["decoder.norm_out.bias"].astype(kind),
        1e-5,
    )
    out = w.linear("decoder.proj_out", x)
    return (
        out.reshape(batch, h, width, 3, 16, 16)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(batch, h * 16, width * 16, 3)
    )


def decode(path, latent, stats):
    w = Weights(path, True)
    if "latents_mean" in w.header:  # P-H3-Image stores its latent statistics in the checkpoint
        stats = {key: w[key].astype(mx.float32) for key in ("latents_mean", "latents_std")}
    z = latent.transpose(1, 2, 0) * mx.array(stats["latents_std"]) + mx.array(
        stats["latents_mean"]
    )
    # Compute in the checkpoint's own dtype: casting 5 GB of bf16 weights to fp16 on every tile batch dominated the stage.
    bf16 = w.header["decoder.x_embedder.weight"]["dtype"] == "BF16"
    z = z.astype(mx.bfloat16 if bf16 else mx.float16)
    yy, th = tile_positions(z.shape[0])
    xx, tw = tile_positions(z.shape[1])
    canvas = np.zeros((z.shape[0] * 16, z.shape[1] * 16, 3), np.float32)
    for i, y in enumerate(yy):
        strip = np.zeros((th * 16, canvas.shape[1], 3), np.float32)
        for start in range(0, len(xx), 4):
            xs = xx[start : start + 4]
            group = mx.stack([z[y : y + th, x : x + tw] for x in xs])
            tiles = np.asarray(decode_tiles(w, group).astype(mx.float32))
            for off, (tile, x) in enumerate(zip(tiles, xs)):
                j = start + off
                tile = tile.copy()
                if i:
                    overlap = (yy[i - 1] + th - y) * 16
                    alpha = (
                        np.arange(overlap, dtype=np.float32)[:, None, None] / overlap
                    )
                    old = canvas[16 * y : 16 * y + overlap, 16 * x : 16 * (x + tw)]
                    tile[:overlap] = old + (tile[:overlap] - old) * alpha
                if j:
                    overlap = (xx[j - 1] + tw - x) * 16
                    alpha = (
                        np.arange(overlap, dtype=np.float32)[None, :, None] / overlap
                    )
                    old = strip[:, 16 * x : 16 * x + overlap]
                    tile[:, :overlap] = old + (tile[:, :overlap] - old) * alpha
                strip[:, 16 * x : 16 * (x + tw)] = tile
        canvas[16 * y : 16 * (y + th)] = strip
    rgb = canvas * np.array([0.229, 0.224, 0.225], np.float32) + np.array(
        [0.485, 0.456, 0.406], np.float32
    )
    if not np.isfinite(rgb).all():
        raise ArithmeticError("Non-finite VAE output")
    return Image.fromarray((rgb.clip(0, 1) * 255).round().astype(np.uint8))
