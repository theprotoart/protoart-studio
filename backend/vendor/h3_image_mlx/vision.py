"""H3's Qwen3-VL visual presentation, including DeepStack and 3-axis text RoPE."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from PIL import Image, ImageOps
from .core import attention, heads, rope


def reference_image(path, max_short_edge):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        short = min(im.size)
        scale = min(max(short, 1024), max_short_edge) / short
        size = tuple(max(32, round(v * scale / 32) * 32) for v in im.size)
        return (
            np.asarray(im.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
            / 255
        )


def visual_features(w, path):
    im = reference_image(path, 1024) * 2 - 1
    gh, gw = im.shape[0] // 16, im.shape[1] // 16
    index = (
        np.arange(gh * gw)
        .reshape(gh // 2, 2, gw // 2, 2)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    pixels = (
        im.reshape(gh, 16, gw, 16, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 768)[index]
    )
    # A still image is repeated for both temporal taps of the patch embedding.
    proj = (
        w["visual.patch_embed.proj.weight"]
        .astype(mx.float32)
        .sum(axis=2)
        .reshape(1152, 768)
    )
    x = (mx.array(pixels) @ proj.T + w["visual.patch_embed.proj.bias"]).astype(
        mx.bfloat16
    )
    yy, xx = np.unravel_index(index, (gh, gw))
    coord = np.stack([yy * 47 / (gh - 1), xx * 47 / (gw - 1)], axis=-1)
    low = np.floor(coord).astype(int)
    frac = coord - low
    table = w["visual.pos_embed.weight"]
    pos = 0
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        indices = np.minimum(low[:, 0] + dy, 47) * 48 + np.minimum(low[:, 1] + dx, 47)
        coef = (frac[:, 0] if dy else 1 - frac[:, 0]) * (
            frac[:, 1] if dx else 1 - frac[:, 1]
        )
        pos = pos + table[mx.array(indices)] * mx.array(coef.astype("f4"))[:, None]
    x = x + pos.astype(x.dtype)
    angle = np.stack([yy, xx], axis=1)[:, :, None] * (10000 ** (-np.arange(18) / 18))
    angles = mx.array(angle.reshape(-1, 36).astype(np.float32))

    def ln(x, p):
        return mx.fast.layer_norm(x, w[p + ".weight"], w[p + ".bias"], 1e-6)

    def merge(x, p, post_shuffle):
        if post_shuffle:
            x = x.reshape(-1, 4608)
        x = ln(x, p + ".norm").reshape(-1, 4608)
        return w.linear(p + ".linear_fc2", nn.gelu(w.linear(p + ".linear_fc1", x)))

    deep = []
    for i in range(27):
        p = f"visual.blocks.{i}"
        q, k, v = (
            heads(t, 72)
            for t in mx.split(
                w.linear(p + ".attn.qkv", ln(x, p + ".norm1")), 3, axis=-1
            )
        )
        x = x + w.linear(
            p + ".attn.proj", attention(rope(q, angles), rope(k, angles), v)
        )
        h = nn.gelu_approx(w.linear(p + ".mlp.linear_fc1", ln(x, p + ".norm2")))
        x = x + w.linear(p + ".mlp.linear_fc2", h)
        if i in (8, 16, 24):
            deep.append(merge(x, f"visual.deepstack_merger_list.{len(deep)}", True))
        mx.eval(x, *deep)
    return merge(x, "visual.merger", False), deep, (gh // 2, gw // 2)


def prepare_prompt(w, tok, prompt, refs):
    pieces = []
    positions = []
    tags = []
    deep = [[], [], []]
    visual_rows = []
    pos = 0
    length = 0

    def add_ids(ids, tag=1):
        nonlocal pos, length
        if ids:
            pieces.append(
                mx.stack([w._read("model.embed_tokens.weight", row=i) for i in ids])
            )
            positions.append(np.tile(np.arange(pos, pos + len(ids))[:, None], (1, 3)))
            tags.extend([tag] * len(ids))
            pos += len(ids)
            length += len(ids)

    for i, path in enumerate(refs, 1):
        add_ids(tok.encode(f"Picture {i}: ", add_special_tokens=False).ids)
        add_ids([151652], 0)
        features, levels, (h, width) = visual_features(w, path)
        pieces.append(features)
        yy, xx = np.indices((h, width))
        positions.append(
            np.stack(
                [np.full(h * width, pos), pos + yy.ravel(), pos + xx.ravel()], axis=1
            )
        )
        visual_rows.extend(range(length, length + h * width))
        length += h * width
        pos += max(h, width)
        tags.extend([0] * (h * width))
        for level, features in zip(deep, levels):
            level.append(features)
        add_ids([151653], 0)
    add_ids(tok.encode(prompt, add_special_tokens=False).ids[:4096])
    coords = np.concatenate(positions)
    axis = np.zeros(64, int)
    axis[1:60:3] = 1
    axis[2:60:3] = 2
    angles = mx.array(
        (coords[:, axis] * 5_000_000 ** (-np.arange(64) / 64)).astype("f4")
    )
    return (
        mx.concatenate(pieces),
        angles,
        mx.array(visual_rows),
        [mx.concatenate(v) for v in deep],
        mx.array(tags),
    )
