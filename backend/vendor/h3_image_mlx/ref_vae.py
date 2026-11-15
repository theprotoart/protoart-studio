"""Encode still reference frames with the H3 VAE's single-frame causal CNN."""

import numpy as np
from .core import mx
from .vae import tile_positions
from .vision import reference_image


def encode_reference(w, path, stats):
    img = reference_image(path, 2048)
    normed = (img - np.array([0.485, 0.456, 0.406], dtype="f4")) / np.array(
        [0.229, 0.224, 0.225], dtype="f4"
    )
    image = mx.array(normed).astype(mx.float16)

    def conv(x, p, stride=1, down=False):
        name = p + ".frame_weight"
        if name not in w.cache:
            # All earlier time taps multiply the causal zero padding on a still frame.
            weight = w[p + ".weight"][:, :, -1].transpose(0, 2, 3, 1)
            w.cache[name] = mx.contiguous(weight)
            mx.eval(w.cache[name])
            w.cache.pop(p + ".weight", None)
        weight = w.cache[name]
        if weight.shape[1] > 1:
            lo = 0 if down else 1
            for axis in (1, 2):
                ids = np.pad(np.arange(x.shape[axis]), (lo, 1), mode="reflect")
                x = mx.take(x, mx.array(ids), axis=axis)
        return mx.conv2d(x, weight, stride=stride) + w[p + ".bias"]

    def norm_act(x, p):
        batch, h, width, c = x.shape
        xx = x.astype(mx.float32).reshape(batch, h, width, 32, c // 32)
        avg = xx.mean((1, 2, 4), keepdims=True)
        xx = (xx - avg) * mx.rsqrt(
            mx.mean(mx.square(xx - avg), axis=(1, 2, 4), keepdims=True) + 1e-6
        )
        yy = (xx.reshape(x.shape) * w[p + ".weight"] + w[p + ".bias"]).astype(x.dtype)
        return yy * mx.sigmoid(yy)

    def moments(tile):
        x = conv(tile, "encoder.conv_in")
        for i in range(6):
            prefix = f"encoder.down.{i}"
            for j in range(2):
                p = f"{prefix}.block.{j}"
                h = conv(norm_act(x, p + ".norm1"), p + ".conv1")
                h = conv(norm_act(h, p + ".norm2"), p + ".conv2")
                if p + ".nin_shortcut.weight" in w.header:
                    x = conv(x, p + ".nin_shortcut")
                x = x + h
            if prefix + ".downsample.conv.weight" in w.header:
                x = conv(x, prefix + ".downsample.conv", stride=2, down=True)
            mx.eval(x)
        x = conv(norm_act(x, "encoder.norm_out"), "encoder.conv_out")
        return conv(x, "quant_conv")

    H, W = image.shape[0] // 16, image.shape[1] // 16
    ys, h = tile_positions(H)
    xs, width = tile_positions(W)
    raw = []
    for y in ys:
        row = []
        for i in range(0, len(xs), 4):
            batch = mx.stack(
                [
                    image[y * 16 : (y + h) * 16, x * 16 : (x + width) * 16]
                    for x in xs[i : i + 4]
                ]
            )
            row.extend(np.asarray(moments(batch).astype(mx.float32)))
        raw.append(row)
    # The encoder blends against each raw preceding tile, then discards its trailing overlap.
    strips = []
    for i, y in enumerate(ys):
        tiles = []
        for j, x in enumerate(xs):
            tile = raw[i][j].copy()
            if i:
                size = ys[i - 1] + h - y
                alpha = np.arange(size, dtype="f4")[:, None, None] / size
                tile[:size] = (1 - alpha) * raw[i - 1][j][-size:] + alpha * tile[:size]
            if j:
                size = xs[j - 1] + width - x
                alpha = np.arange(size, dtype="f4")[None, :, None] / size
                tile[:, :size] = (1 - alpha) * raw[i][j - 1][:, -size:] + alpha * tile[
                    :, :size
                ]
            tiles.append(
                tile[
                    : ys[i + 1] - y if i + 1 < len(ys) else h,
                    : xs[j + 1] - x if j + 1 < len(xs) else width,
                ]
            )
        strips.append(np.concatenate(tiles, axis=1))
    mean, logvar = np.split(np.concatenate(strips, axis=0), 2, axis=-1)
    noise = (
        np.random.default_rng(42)
        .standard_normal((24, H, W), dtype=np.float32)
        .transpose(1, 2, 0)
    )
    z = mean + np.exp(0.5 * logvar.clip(-30, 20)) * noise
    z = (z - np.array(stats["latents_mean"], dtype="f4")) / np.array(
        stats["latents_std"], dtype="f4"
    )
    return mx.array(z.transpose(2, 0, 1))
