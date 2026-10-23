"""SD 1.5 / SDXL UNet on MLX (Apple's own array library for Apple chips), read from diffusers-named weights.

A plain functional version: the weights stay a dict of MLX arrays (loaded lazily, memory-mapped, like DiffusionBee's
engine), activations are NHWC, attention uses MLX's fused scaled_dot_product_attention (Metal FlashAttention-style),
and the prompt's cross-attention keys/values are worked out once per picture (see `prepare`).
Written from the published diffusers UNet2DConditionModel architecture (Apache-2.0); no DiffusionBee code.
"""
import math

import mlx.core as mx


class UNet:
    def __init__(self, weights, cfg, bits=None, group_size=64, attn_mb=0):
        """weights: {diffusers name: mx.array (torch layout)}; cfg: the diffusers unet config.json dict.
        bits=8 or 4 keeps the big linear layers compressed (unpacked on the GPU as they're used)."""
        self.cfg = cfg
        self.dtype = mx.float16
        self.w = {}
        self.q = {}  # name -> (packed weight, scales, biases)
        self.bits, self.group_size = bits, group_size
        self.attn_mb = attn_mb
        for k in list(weights):
            v = weights.pop(k)
            if v.ndim == 4:  # conv OIHW -> OHWI
                v = v.transpose(0, 2, 3, 1)
            if v.dtype != self.dtype and v.dtype in (mx.float32, mx.bfloat16):
                v = v.astype(self.dtype)
            if bits and v.ndim == 2 and k.endswith(".weight") and v.shape[1] % group_size == 0 and v.shape[0] >= 256:
                self.q[k[: -len(".weight")]] = mx.quantize(v, group_size=group_size, bits=bits)
                mx.eval(self.q[k[: -len(".weight")]])
                self.w[k] = mx.zeros((v.shape[0], 0), dtype=self.dtype)  # placeholder: keeps the shape lookups working
                continue
            self.w[k] = v
        heads = cfg.get("attention_head_dim", 8)
        nb = len(cfg["block_out_channels"])
        self.heads = list(heads) if isinstance(heads, (list, tuple)) else [heads] * nb
        # SDXL gives head *dim* through num_attention_heads=None + attention_head_dim list = number of heads
        self.linear_proj = bool(cfg.get("use_linear_projection"))
        tl = cfg.get("transformer_layers_per_block", 1)
        self.tlayers = list(tl) if isinstance(tl, (list, tuple)) else [tl] * nb
        self.kv = {}

    # ---------- building blocks ----------
    def _lin(self, x, name, bias=True):
        q = self.q.get(name)
        if q is not None:
            y = mx.quantized_matmul(x, *q, transpose=True, group_size=self.group_size, bits=self.bits)
        else:
            y = x @ self.w[name + ".weight"].T
        b = self.w.get(name + ".bias") if bias else None
        return y + b if b is not None else y

    def _conv(self, x, name, stride=1):
        wt = self.w[name + ".weight"]
        pad = wt.shape[1] // 2
        y = mx.conv2d(x, wt, stride=stride, padding=pad)
        b = self.w.get(name + ".bias")
        return y + b if b is not None else y

    def _gn(self, x, name, groups=32, eps=1e-5):
        b, h, w, c = x.shape
        xf = x.astype(mx.float32).reshape(b, h * w, groups, c // groups)
        mean = xf.mean(axis=(1, 3), keepdims=True)
        var = xf.var(axis=(1, 3), keepdims=True)
        xf = ((xf - mean) * mx.rsqrt(var + eps)).reshape(b, h, w, c).astype(x.dtype)
        return xf * self.w[name + ".weight"] + self.w[name + ".bias"]

    def _ln(self, x, name, eps=1e-5):
        return mx.fast.layer_norm(x, self.w[name + ".weight"], self.w[name + ".bias"], eps)

    def _resnet(self, x, temb, name):
        h = self._conv(nn_silu(self._gn(x, name + ".norm1")), name + ".conv1")
        h = h + self._lin(nn_silu(temb), name + ".time_emb_proj")[:, None, None, :]
        h = self._conv(nn_silu(self._gn(h, name + ".norm2")), name + ".conv2")
        if name + ".conv_shortcut.weight" in self.w:
            x = self._conv(x, name + ".conv_shortcut")
        return x + h

    def _attn(self, x, name, heads, ctx=None):
        b, n, c = x.shape
        q = self._lin(x, name + ".to_q", bias=False).reshape(b, n, heads, -1).transpose(0, 2, 1, 3)
        if ctx is None:
            k = self._lin(x, name + ".to_k", bias=False).reshape(b, n, heads, -1).transpose(0, 2, 1, 3)
            v = self._lin(x, name + ".to_v", bias=False).reshape(b, n, heads, -1).transpose(0, 2, 1, 3)
        else:
            k, v = self.kv[name]
        o = self._sdpa(q, k, v)
        o = o.transpose(0, 2, 1, 3).reshape(b, n, c)
        return self._lin(o, name + ".to_out.0")

    def _sdpa(self, q, k, v):
        """Attention in query chunks small enough that the score table stays under `attn_mb` (like attention
        slicing in PyTorch: on 8 GB Macs the full table for a big picture pushes the GPU into swap)."""
        scale = 1.0 / math.sqrt(q.shape[-1])
        b, h, n, _d = q.shape
        per_row = b * h * k.shape[2] * 2 * 3  # scores + softmax + a spare, fp16
        chunk = max(256, int(self.attn_mb * 1e6 // per_row))
        if not self.attn_mb or chunk >= n:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        outs = []
        for i in range(0, n, chunk):
            o = mx.fast.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v, scale=scale)
            mx.eval(o)  # finish (and free) this chunk's table before the next
            outs.append(o)
        return mx.concatenate(outs, axis=2)

    def _transformer(self, x, name, heads, layers):
        b, h, w, c = x.shape
        res = x
        x = self._gn(x, name + ".norm", eps=1e-6)
        if self.linear_proj:
            x = self._lin(x.reshape(b, h * w, c), name + ".proj_in")
        else:
            x = self._conv(x, name + ".proj_in").reshape(b, h * w, c)
        for i in range(layers):
            t = f"{name}.transformer_blocks.{i}"
            x = x + self._attn(self._ln(x, t + ".norm1"), t + ".attn1", heads)
            x = x + self._attn(self._ln(x, t + ".norm2"), t + ".attn2", heads, ctx=True)
            y = self._lin(self._ln(x, t + ".norm3"), t + ".ff.net.0.proj")
            a, gate = mx.split(y, 2, axis=-1)
            x = x + self._lin(a * nn_gelu(gate), t + ".ff.net.2")
        if self.linear_proj:
            x = self._lin(x, name + ".proj_out").reshape(b, h, w, c)
        else:
            x = self._conv(x.reshape(b, h, w, c), name + ".proj_out")
        return x + res

    # ---------- per picture ----------
    def prepare(self, ctx):
        """Work out every cross-attention key/value from the prompt once (DiffusionBee's 'fixed' part)."""
        self.kv = {}
        ctx = ctx.astype(self.dtype)
        b = ctx.shape[0]
        for k in self.w:
            if k.endswith(".attn2.to_k.weight"):
                name = k[: -len(".to_k.weight")]
                heads = self._heads_for(name)
                hd = self.w[name + ".to_k.weight"].shape[0] // heads  # (placeholder keeps dim 0)
                kk = self._lin(ctx, name + ".to_k", bias=False).reshape(b, -1, heads, hd).transpose(0, 2, 1, 3)
                vv = self._lin(ctx, name + ".to_v", bias=False).reshape(b, -1, heads, hd).transpose(0, 2, 1, 3)
                self.kv[name] = (kk, vv)
        mx.eval([t for p in self.kv.values() for t in p])

    def _heads_for(self, name):
        if name.startswith("mid_block"):
            return self.heads[-1]
        part = name.split(".")
        i = int(part[1])
        if part[0] == "down_blocks":
            return self.heads[i]
        return list(reversed(self.heads))[i]

    def _layers_for(self, kind, i):
        return self.tlayers[i] if kind == "down" else list(reversed(self.tlayers))[i]

    def time_embedding(self, t):
        dim = self.cfg["block_out_channels"][0]
        half = dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = mx.array(t, dtype=mx.float32).reshape(-1, 1) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # flip_sin_to_cos
        return emb.astype(self.dtype)

    def __call__(self, x, t, added=None):
        """x: (B, H, W, C) latents NHWC; t: timestep (scalar or (B,)); added: SDXL (text_embeds, time_ids)."""
        b = x.shape[0]
        x = x.astype(self.dtype)
        te = self.time_embedding(t)
        if te.shape[0] != b:
            te = mx.broadcast_to(te, (b, te.shape[1]))
        temb = self._lin(nn_silu(self._lin(te, "time_embedding.linear_1")), "time_embedding.linear_2")
        if added is not None:  # SDXL: pooled text + size/crop conditioning
            text_embeds, time_ids = added
            tid = self._time_ids_embedding(time_ids)
            aug = mx.concatenate([text_embeds.astype(self.dtype), tid], axis=-1)
            temb = temb + self._lin(nn_silu(self._lin(aug, "add_embedding.linear_1")), "add_embedding.linear_2")
        h = self._conv(x, "conv_in")
        skips = [h]
        nb = len(self.cfg["block_out_channels"])
        for i in range(nb):
            n = f"down_blocks.{i}"
            j = 0
            while f"{n}.resnets.{j}.conv1.weight" in self.w:
                h = self._resnet(h, temb, f"{n}.resnets.{j}")
                if f"{n}.attentions.{j}.proj_in.weight" in self.w:
                    h = self._transformer(h, f"{n}.attentions.{j}", self.heads[i], self._layers_for("down", i))
                skips.append(h)
                j += 1
            if f"{n}.downsamplers.0.conv.weight" in self.w:
                h = self._conv(h, f"{n}.downsamplers.0.conv", stride=2)
                skips.append(h)
        h = self._resnet(h, temb, "mid_block.resnets.0")
        h = self._transformer(h, "mid_block.attentions.0", self.heads[-1], self.tlayers[-1])
        h = self._resnet(h, temb, "mid_block.resnets.1")
        for i in range(nb):
            n = f"up_blocks.{i}"
            j = 0
            while f"{n}.resnets.{j}.conv1.weight" in self.w:
                h = mx.concatenate([h, skips.pop()], axis=-1)
                h = self._resnet(h, temb, f"{n}.resnets.{j}")
                if f"{n}.attentions.{j}.proj_in.weight" in self.w:
                    h = self._transformer(h, f"{n}.attentions.{j}", self._heads_for(f"{n}.x"), self._layers_for("up", i))
                j += 1
            if f"{n}.upsamplers.0.conv.weight" in self.w:
                bb, hh, ww, cc = h.shape
                h = mx.broadcast_to(h[:, :, None, :, None, :], (bb, hh, 2, ww, 2, cc)).reshape(bb, hh * 2, ww * 2, cc)
                h = self._conv(h, f"{n}.upsamplers.0.conv")
        h = self._conv(nn_silu(self._gn(h, "conv_norm_out")), "conv_out")
        return h

    def _time_ids_embedding(self, time_ids):
        dim = self.cfg.get("addition_time_embed_dim", 256)
        half = dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        b = time_ids.shape[0]
        args = time_ids.astype(mx.float32).reshape(-1, 1) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return emb.reshape(b, -1).astype(self.dtype)


def nn_silu(x):
    return x * mx.sigmoid(x)


def nn_gelu(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))
