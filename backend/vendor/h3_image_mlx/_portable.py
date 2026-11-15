"""Evaluate H3's quantized projections on Apple GPUs and the MLX CPU backend."""

import platform
from functools import cache
from pathlib import Path

import mlx.core as mx

# An int8 dot product of length 1024 has an absolute sum <= 2**24, so its
# integer partial sums are exactly representable in float32. H3's longest
# projection has K=25600, safely within int32 for the combined sum.
_DOT_CHUNK = 1024
_TILE = 1024
_CHUNKS_PER_BATCH = 4


@cache
def _simd_gemm():
    return mx.fast.metal_kernel(
        name="h3_portable_simd",
        input_names=["x", "w", "s", "z"],
        output_names=["y"],
        header="#include <metal_simdgroup_matrix>\n",
        source=Path(__file__).with_name("kernels").joinpath("portable_simd.metal").read_text(),
    )


@cache
def _direct_tensor_kernel():
    kernels = Path(__file__).with_name("kernels")
    return mx.fast.metal_kernel(
        name="h3_portable_tensor_i8",
        input_names=["x", "w"],
        output_names=["y"],
        header=(kernels / "mpp_tensor_ops.metal").read_text(),
        source=(kernels / "portable_tensor.metal").read_text(),
    )


def _tensor_dot(x, weight):
    rows, _ = x.shape
    channels = weight.shape[0]
    # TensorOps may access a complete final tile on a physical GPU.
    tail = (-rows) % 64
    if tail:
        x = mx.pad(x, [(0, tail), (0, 0)])
    return _direct_tensor_kernel()(
        inputs=[x, weight], template=[("BM", 64), ("BN", 64)],
        grid=((channels + 63) // 64 * 128, (rows + 63) // 64, 1),
        threadgroup=(128, 1, 1), output_shapes=[(rows + tail, channels)], output_dtypes=[mx.int32],
    )[0][:rows]


@cache
def _has_integer_tensor_ops():
    version = platform.mac_ver()[0]
    if not version or tuple(map(int, version.split(".")[:2])) < (26, 4):
        return False
    # MLX can construct a kernel lazily. Evaluate a tail tile before selecting
    # this path, since shader compilation and device support vary independently.
    try:
        x = mx.full((65, 64), -128, dtype=mx.int8)
        w = mx.full((64, 64), 127, dtype=mx.int8)
        return mx.all(_tensor_dot(x, w) == -128 * 127 * 64).item()
    except RuntimeError:
        return False


def portable_matmul(xq, weight, xs, scale, output_dtype):
    """Compute row-scaled int8 dot products without the M5 Metal TensorOps kernel."""
    rows, width = xq.shape
    channels = weight.shape[0]
    row_scale = xs.reshape(rows, 1).astype(mx.float32)
    channel_scale = mx.broadcast_to(scale.reshape(-1), (channels,)).astype(mx.float32)

    aligned = width % 64 == 0 and channels % 64 == 0
    on_gpu = mx.default_device() == mx.Device(mx.gpu)
    if rows > 512 and aligned and on_gpu and _has_integer_tensor_ops():
        # Bound the intermediate int32 output for high-resolution generation.
        chunks = []
        for begin in range(0, rows, 1536):
            tile = slice(begin, begin + 1536)
            dots = _tensor_dot(xq[tile], weight)
            result = (dots.astype(mx.float32) * row_scale[tile] * channel_scale).astype(output_dtype)
            if rows > 1536:
                mx.eval(result)
            chunks.append(result)
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks)
    if rows <= 512 and aligned and on_gpu:
        bm, bn, bk = (32, 64, 64) if rows <= 128 else (64, 64, 32)
        return _simd_gemm()(
            inputs=[xq, weight, row_scale, channel_scale],
            template=[("T", output_dtype), ("BM", bm), ("BN", bn), ("BK", bk)],
            grid=(channels // bn * 128, (rows + bm - 1) // bm, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, channels)],
            output_dtypes=[output_dtype],
        )[0]

    # Keep medium-length sequences together so a short trailing row tile
    # does not reload every weight. Bound both the row count and dot planes.
    full_rows = (
        512 < rows <= 1536 and aligned and _DOT_CHUNK <= width <= 8192
        and mx.default_device() == mx.Device(mx.gpu)
    )
    row_height = rows if full_rows else _TILE
    batch_width = width if full_rows else _CHUNKS_PER_BATCH * _DOT_CHUNK
    full_width = width - width % _DOT_CHUNK

    row_tiles = []
    for r in range(0, rows, row_height):
        columns = []
        row_slice = slice(r, r + row_height)
        a = xq[row_slice]
        if full_rows:
            a = a.astype(mx.float32)
        for c in range(0, channels, _TILE):
            col_slice = slice(c, c + _TILE)
            # Batch exact partial dots along K. Limit the number of temporary
            # result planes, even for H3's 25,600-wide text projections.
            b = weight[col_slice]
            total = None
            for k in range(0, full_width, batch_width):
                inner = slice(k, min(k + batch_width, full_width))
                av = a[:, inner].reshape(a.shape[0], -1, _DOT_CHUNK).transpose(1, 0, 2)
                bv = b[:, inner].reshape(b.shape[0], -1, _DOT_CHUNK).transpose(1, 0, 2)
                dots = av.astype(mx.float32) @ bv.astype(mx.float32).swapaxes(1, 2)
                partial = dots.astype(mx.int32).sum(axis=0)
                total = partial if total is None else total + partial
            if full_width < width:
                tail = a[:, full_width:].astype(mx.float32) @ b[:, full_width:].astype(mx.float32).T
                partial = tail.astype(mx.int32)
                total = partial if total is None else total + partial
            # Evaluate before advancing to the next tile: no dense floating
            # point copy of an entire large weight is retained across tiles.
            y = total.astype(mx.float32) * row_scale[row_slice] * channel_scale[col_slice]
            y = y.astype(output_dtype)
            mx.eval(y)
            columns.append(y)
        row = mx.concatenate(columns, axis=1) if len(columns) > 1 else columns[0]
        mx.eval(row)
        row_tiles.append(row)
    return mx.concatenate(row_tiles, axis=0) if len(row_tiles) > 1 else row_tiles[0]
