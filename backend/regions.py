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


def area_boxes(layout, ratios, n, boxes=None):
    """The areas as (x0, y0, x1, y1) in 0..1: the boxes you drew (layout "free", 📦 Boxes) or the columns / rows / grid."""
    if layout == "free":
        out = []
        for b in list(boxes or [])[:n]:
            try:
                x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in b)
            except (TypeError, ValueError):
                x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
            out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        return out + [(0.0, 0.0, 1.0, 1.0)] * (n - len(out))
    return parse_layout(layout or "cols", ratios or "", n)


def _grid(box):
    """(x0, y0, x1, y1) in 0..1 → [y_min, x_min, y_max, x_max] on the 0-1000 grid (FLUX 3's and Ideogram 4's boxes)."""
    x0, y0, x1, y1 = box
    return [round(y0 * 1000), round(x0 * 1000), round(y1 * 1000), round(x1 * 1000)]


def _quoted(text):
    """(words to show, the rest) when an area's prompt has words in quotes ('SALE' / "SALE"), else (None, text)."""
    import re
    m = re.search(r"[\"“”']([^\"“”']{1,200})[\"“”']", text)
    return (m.group(1).strip(), text) if m else (None, text)


def _areas(layout):
    parts = [p.strip() if isinstance(p, str) else "" for p in (layout.get("prompts") or [])]
    boxes = area_boxes(layout.get("layout"), layout.get("ratios"), len(parts), layout.get("boxes"))
    return [(t, b) for t, b in zip(parts, boxes) if t]


def layout_prompt(family, prompt, layout, width=1024, height=1024):
    """The prompt with the areas written in the way each model reads a layout (empty areas skipped; no areas: the
    prompt as it was): FLUX 3 → flux3_boxes, Ideogram 4 → its JSON caption, MOSS-Image → its JSON layout."""
    areas = _areas(layout or {})
    if not areas:
        return prompt
    if family == "ideogram4":
        return ideogram4_caption(prompt, areas)
    if family == "moss":
        return moss_layout(prompt, areas, width, height)
    return flux3_boxes(prompt, layout.get("layout"), layout.get("ratios"), layout.get("prompts") or [], layout.get("boxes"))


def ideogram4_caption(prompt, areas):
    """Ideogram 4's JSON caption: the whole picture as high_level_description, each area an element (text when its
    prompt has words in quotes) with its bbox on the 0-1000 grid. Key order as Ideogram's prompting guide."""
    import json
    elements = []
    for text, box in areas:
        words, desc = _quoted(text)
        el = {"type": "text" if words else "obj", "bbox": _grid(box)}
        if words:
            el["text"] = words
        el["desc"] = desc
        elements.append(el)
    cap = {"high_level_description": (prompt or "").strip(),
           "compositional_deconstruction": {"background": "", "elements": elements}}
    return json.dumps(cap, ensure_ascii=False)


def moss_layout(prompt, areas, width, height):
    """MOSS-Image's JSON layout prompt: summary caption plus one item per area (label object, or text with the words
    in quotes), bbox [y_min, x_min, y_max, x_max] in pixels from the picture's centre ("abs_centralized", "yxyx")."""
    import json
    import math
    hx, hy = int(width) // 2, int(height) // 2
    items = []
    for text, (x0, y0, x1, y1) in areas:
        words, _ = _quoted(text)
        item = {"label": "text" if words else "object", "caption": text}
        if words:
            item["text"] = words
        item["bbox"] = [math.floor(round(y0 * height, 9)) - hy, math.floor(round(x0 * width, 9)) - hx,
                        math.ceil(round(y1 * height, 9)) - hy, math.ceil(round(x1 * width, 9)) - hx]
        items.append(item)
    out = {"summary": {"caption": (prompt or "").strip()}, "items": items,
           "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def flux3_boxes(prompt, layout, ratios, prompts, boxes=None):
    """🗺 Regions for FLUX 3 Image Klein, which reads layout boxes straight from the prompt: each area becomes an
    element <obj_N> with its box ([y_min, x_min, y_max, x_max] on a 0-1000 grid) and its description, listed as JSON
    after the main prompt (Black Forest Labs' format, as ComfyUI's FLUX.3 prompt builders write it). Empty areas are
    skipped."""
    import json
    parts = [p.strip() if isinstance(p, str) else "" for p in prompts]
    boxes = area_boxes(layout, ratios, len(parts), boxes)
    items, tags = [], []
    for i, (text, box) in enumerate(zip(parts, boxes), 1):
        if not text:
            continue
        rid = f"obj_{i}"
        items.append({"id": rid, "bbox": _grid(box), "desc": text})
        tags.append(f"{text} <{rid}>")
    if not items:
        return prompt
    main = (prompt or "").strip()
    if main and main[-1] not in ".!?":
        main += "."
    lead = (main + " " if main else "") + "; ".join(tags) + "."
    return lead + " " + json.dumps(items, ensure_ascii=False)


def flux3_edit_boxes(prompt, items):
    """📦 Move / remove / add with boxes (✨ Edit with words with FLUX 3): each item is {"desc", "act": "move" |
    "remove" | "add", "src": box where it is now in your picture, "tgt": box where it goes} (boxes x0, y0, x1, y1 in
    0..1). Written as BFL's edit layout (as ComfyUI's FLUX.3 prompt builder does with a reference picture): a sentence
    per item citing <obj_N>, then rows {id, from, src_bbox, tgt_bbox, desc}: from "ref_image_0" (your picture) when it
    has a source box, "target" when it's new. A remove has no target box; an add has no source box."""
    import json

    def box(b):
        try:
            x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in b)
        except (TypeError, ValueError):
            return None
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            return None
        return _grid((x0, y0, x1, y1))

    rows, says = [], []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        desc = str(it.get("desc") or "").strip()
        act = it.get("act") if it.get("act") in ("move", "remove", "add") else "move"
        src = box(it.get("src")) if act != "add" else None
        tgt = box(it.get("tgt")) if act != "remove" else None
        if not desc or (act != "add" and not src) or (act != "remove" and not tgt):
            continue
        rid = f"obj_{len(rows) + 1}"
        rows.append({"id": rid, "from": "ref_image_0" if src else "target", "src_bbox": src, "tgt_bbox": tgt, "desc": desc})
        says.append({"move": f"Move the {desc} <{rid}> to its new place", "remove": f"Remove the {desc} <{rid}> and fill in what was behind it",
                     "add": f"Add {desc} <{rid}>"}[act])
    if not rows:
        return prompt
    main = (prompt or "").strip()
    if main and main[-1] not in ".!?":
        main += "."
    lead = (main + " " if main else "") + "; ".join(says) + ". Keep everything else exactly as it is."
    return lead + " " + json.dumps(rows, ensure_ascii=False)
