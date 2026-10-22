"""Regional prompts (like A1111's Regional Prompter / Latent Couple), for SD 1.5 and SDXL.

The picture is split into areas (columns, rows, or a grid such as "1,1;1,2"), and each area gets its own prompt:
"a red-haired knight" on the left, "a blue dragon" on the right. Inside the model's cross-attention layers each
area's pixels look only at their own prompt (plus a little of the main prompt, so the picture stays one scene).

Works by swapping in an attention processor for the cross-attention layers while the picture is made; they're put
back afterwards.
"""
import math


def parse_layout(layout, ratios, n):
    """-> list of n boxes (x0, y0, x1, y1) in 0..1.
    layout "cols": side by side; "rows": stacked; "grid": ratios like "1,1;2,1" (rows separated by ';')."""
    def split(parts):
        parts = [max(0.01, float(p)) for p in parts]
        total = sum(parts)
        out, acc = [], 0.0
        for p in parts:
            out.append((acc / total, (acc + p) / total))
            acc += p
        return out

    if layout == "grid" and isinstance(ratios, str) and ";" in ratios:
        rows = [r for r in ratios.split(";") if r.strip()]
        heights = split([1] * len(rows))
        boxes = []
        for (y0, y1), row in zip(heights, rows):
            for x0, x1 in split([c for c in row.split(",") if c.strip()]):
                boxes.append((x0, y0, x1, y1))
        return boxes[:n] + [(0, 0, 1, 1)] * max(0, n - len(boxes))
    if isinstance(ratios, str):
        ratios = [r for r in ratios.replace(";", ",").split(",") if r.strip()]
    ratios = list(ratios or []) or [1] * n
    ratios = (ratios + [1] * n)[:n]
    spans = split(ratios)
    if layout == "rows":
        return [(0.0, a, 1.0, b) for a, b in spans]
    return [(a, 0.0, b, 1.0) for a, b in spans]


def masks(boxes, h, w, device, dtype):
    """Each region's mask at the latent size (h × w), with a soft 1-latent-pixel edge; together they sum to 1."""
    import torch
    out = []
    for x0, y0, x1, y1 in boxes:
        m = torch.zeros(h, w)
        ya, yb = int(round(y0 * h)), max(int(round(y0 * h)) + 1, int(round(y1 * h)))
        xa, xb = int(round(x0 * w)), max(int(round(x0 * w)) + 1, int(round(x1 * w)))
        m[ya:yb, xa:xb] = 1.0
        out.append(m)
    stack = torch.stack(out)
    if stack.shape[1] > 4 and stack.shape[2] > 4:  # soften the borders a touch
        k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
        stack = torch.nn.functional.conv2d(stack[:, None], k[None, None], padding=1)[:, 0]
    stack = stack / stack.sum(0, keepdim=True).clamp(min=1e-6)
    return stack.to(device, dtype)


class RegionalAttn:
    """Cross-attention processor: the "cond" half of the batch attends per region; the "uncond" half as normal."""

    def __init__(self, region_embeds, region_masks, latent_hw, base_weight=0.2, cfg=True):
        self.embeds = region_embeds      # list of [1, L, D] tensors (one per region)
        self.masks = region_masks        # [R, h, w] at latent size
        self.lh, self.lw = latent_hw
        self.base = float(base_weight)
        self.cfg = cfg
        self._cache = {}

    def _mask_for(self, hw):
        if hw not in self._cache:
            import torch
            f = math.sqrt(self.lh * self.lw / hw)
            h, w = max(1, round(self.lh / f)), max(1, round(self.lw / f))
            if h * w != hw:  # odd sizes: find the factor pair closest to the latent's shape
                h = max(1, round(math.sqrt(hw * self.lh / self.lw)))
                w = hw // h
            m = torch.nn.functional.interpolate(self.masks[:, None].float(), size=(h, w), mode="bilinear",
                                                align_corners=False)[:, 0]
            m = m / m.sum(0, keepdim=True).clamp(min=1e-6)
            self._cache[hw] = m.reshape(m.shape[0], h * w, 1).to(self.masks.dtype)
        return self._cache[hw]

    @staticmethod
    def _attend(attn, q, ctx):
        import torch.nn.functional as F
        b = q.shape[0]
        k = attn.to_k(ctx)
        v = attn.to_v(ctx)
        heads = attn.heads
        hd = k.shape[-1] // heads
        q_ = q.view(b, -1, heads, hd).transpose(1, 2)
        k_ = k.view(k.shape[0], -1, heads, hd).transpose(1, 2)
        v_ = v.view(v.shape[0], -1, heads, hd).transpose(1, 2)
        if k_.shape[0] != b:
            k_, v_ = k_.expand(b, -1, -1, -1), v_.expand(b, -1, -1, -1)
        o = F.scaled_dot_product_attention(q_, k_, v_)
        return o.transpose(1, 2).reshape(b, -1, heads * hd)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        if encoder_hidden_states is None:  # self-attention: nothing regional about it
            from diffusers.models.attention_processor import AttnProcessor2_0
            return AttnProcessor2_0()(attn, hidden_states, None, attention_mask, temb)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        q = attn.to_q(hidden_states)
        base = self._attend(attn, q, encoder_hidden_states)  # the whole prompt, everywhere
        bsz, hw = hidden_states.shape[0], hidden_states.shape[1]
        cond = slice(bsz // 2, bsz) if self.cfg and bsz >= 2 else slice(0, bsz)
        qc = q[cond]
        m = self._mask_for(hw)
        regional = 0
        for r, emb in enumerate(self.embeds):
            regional = regional + self._attend(attn, qc, emb.to(qc.dtype)) * m[r]
        out = base.clone()
        out[cond] = self.base * base[cond] + (1 - self.base) * regional
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if attn.residual_connection:
            out = out + hidden_states
        return out / attn.rescale_output_factor


def install(unet, processor):
    """Use `processor` for every cross-attention layer. -> the original processors (to put back)."""
    original = dict(unet.attn_processors)
    new = {name: (processor if name.endswith("attn2.processor") else proc) for name, proc in original.items()}
    unet.set_attn_processor(new)
    return original


def remove(unet, original):
    unet.set_attn_processor(original)
