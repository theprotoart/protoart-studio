"""Work out the prompt's part of cross-attention once per picture instead of at every step (SD 1.5 / SDXL).

In every cross-attention layer the UNet turns the prompt into "keys" and "values" (to_k / to_v). The prompt doesn't
change between steps, so those are the same at every step: this keeps them from the first step and reuses them.
DiffusionBee's engine does the same (its "UNetXLFixed" part). The picture comes out exactly the same.

Only the cross-attention layers (attn2) are changed; self-attention keeps whatever the app set (sliced or not).
Layers with anything unusual (an attention mask, an image-prompt adapter, regional prompts…) are left alone, and the
cache is emptied after every picture, so a new prompt, LoRA or strength is always worked out fresh.
"""

_PLAIN = ("AttnProcessor2_0", "AttnProcessor", "SlicedAttnProcessor", "AttnProcessorNPU")


class CachedCrossAttn:
    def __init__(self, base):
        self.base = base
        self.src = None   # the prompt tensor the keys/values were made from (held, so its memory can't be reused)
        self.kv = None
        self.hits = 0

    def clear(self):
        self.src = self.kv = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **kw):
        if (encoder_hidden_states is None or attention_mask is not None or a or kw or hidden_states.ndim != 3
                or getattr(attn, "spatial_norm", None) is not None or getattr(attn, "group_norm", None) is not None
                or getattr(attn, "norm_cross", None) or getattr(attn, "residual_connection", False)
                or getattr(attn, "norm_q", None) is not None or getattr(attn, "norm_k", None) is not None):
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *a, **kw)
        import torch.nn.functional as F
        b = hidden_states.shape[0]
        heads = attn.heads
        q = attn.to_q(hidden_states)
        if self.src is encoder_hidden_states and self.kv is not None and self.kv[0].shape[0] == b:
            k, v = self.kv
            self.hits += 1
        else:
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            hd = k.shape[-1] // heads
            k = k.view(b, -1, heads, hd).transpose(1, 2)
            v = v.view(b, -1, heads, hd).transpose(1, 2)
            self.src, self.kv = encoder_hidden_states, (k, v)
        hd = q.shape[-1] // heads
        q = q.view(b, -1, heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=getattr(attn, "scale", None))
        out = out.transpose(1, 2).reshape(b, -1, heads * hd).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if getattr(attn, "rescale_output_factor", 1.0) != 1.0:
            out = out / attn.rescale_output_factor
        return out


def install(unet):
    """Put the cache on every plain cross-attention layer. Returns how many layers use it (0 = nothing changed)."""
    procs = getattr(unet, "attn_processors", None)
    if not procs:
        return 0
    new, n = {}, 0
    for name, p in procs.items():
        if isinstance(p, CachedCrossAttn):
            p.clear()
            new[name] = p
            n += 1
        elif ".attn2." in name and type(p).__name__ in _PLAIN:
            new[name] = CachedCrossAttn(p)
            n += 1
        else:
            new[name] = p
    if n:
        unet.set_attn_processor(new)
    return n


def clear(unet):
    """Empty the caches (after each picture) without taking the layers out."""
    for p in (getattr(unet, "attn_processors", None) or {}).values():
        if isinstance(p, CachedCrossAttn):
            p.clear()


def remove(unet):
    """Put the original layers back (needed before anything that swaps attention layers, e.g. image prompts)."""
    procs = getattr(unet, "attn_processors", None) or {}
    if any(isinstance(p, CachedCrossAttn) for p in procs.values()):
        unet.set_attn_processor({k: (p.base if isinstance(p, CachedCrossAttn) else p) for k, p in procs.items()})


def hits(unet):
    return sum(p.hits for p in (getattr(unet, "attn_processors", None) or {}).values() if isinstance(p, CachedCrossAttn))
