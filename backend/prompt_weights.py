"""Long prompts and word weights for SD 1.5 and SDXL, the way A1111 / Draw Things / Civitai prompts use them.

  (word)        x1.1        ((word)) x1.21 ...
  (word:1.4)    x1.4        (word:0.6) weaker
  [word]        /1.1
  \\( \\)        a literal bracket
  BREAK         start a new 75-token chunk here

CLIP reads 75 tokens at a time. Longer prompts are split into chunks of 75, each encoded on its own and
joined, so nothing gets cut off. Weights scale each token's embedding, then the chunk is rescaled to keep
its original average (A1111's method), so weights change emphasis without washing out the image.
"""
import re

_re_attention = re.compile(r"""
\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:
""", re.X)
_re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)
_re_needs = re.compile(r"[()\[\]]|\bBREAK\b")
CHUNK = 75


def parse(text):
    """'a (cat:1.3) on [grass]' -> [['a ', 1.0], ['cat', 1.3], [' on ', 1.0], ['grass', 0.909]]
    A [ 'BREAK', -1 ] entry marks a forced chunk break."""
    res, round_b, square_b = [], [], []

    def multiply(start, m):
        for p in range(start, len(res)):
            res[p][1] *= m

    for m in _re_attention.finditer(text):
        t, w = m.group(0), m.group(1)
        if t.startswith("\\"):
            res.append([t[1:], 1.0])
        elif t == "(":
            round_b.append(len(res))
        elif t == "[":
            square_b.append(len(res))
        elif w is not None and round_b:
            try:
                multiply(round_b.pop(), float(w))
            except ValueError:
                res.append([t, 1.0])
        elif t == ")" and round_b:
            multiply(round_b.pop(), 1.1)
        elif t == "]" and square_b:
            multiply(square_b.pop(), 1 / 1.1)
        else:
            for i, part in enumerate(_re_break.split(t)):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])
    for p in round_b:
        multiply(p, 1.1)
    for p in square_b:
        multiply(p, 1 / 1.1)
    if not res:
        res = [["", 1.0]]
    i = 0
    while i + 1 < len(res):  # merge neighbours with the same weight
        if res[i][1] == res[i + 1][1] and res[i][1] != -1:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def needs_custom(prompt, negative, tokenizer):
    """True when the plain pipeline would cut the prompt off or ignore weights."""
    for p in (prompt or "", negative or ""):
        if _re_needs.search(p):
            return True
        try:
            if len(tokenizer(p, add_special_tokens=False, truncation=False).input_ids) > CHUNK:
                return True
        except Exception:
            pass
    return False


def strip_weights(text):
    """Plain text without weight syntax (for places that only need words, e.g. file names)."""
    return "".join(t for t, w in parse(text) if w != -1)


def tokenize_chunks(pipe, tokenizer, text):
    """-> list of chunks; each chunk = (ids[77], weights[77])."""
    convert = getattr(pipe, "maybe_convert_prompt", None)  # expands multi-vector textual inversion tokens
    tokens, weights = [], []
    chunks = []

    def close():
        nonlocal tokens, weights
        ids = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        ws = [1.0] + weights + [1.0]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids += [pad] * (CHUNK + 2 - len(ids))
        ws += [1.0] * (CHUNK + 2 - len(ws))
        chunks.append((ids, ws))
        tokens, weights = [], []

    for frag, w in parse(text):
        if w == -1:
            close()
            continue
        if convert is not None:
            try:
                frag = convert(frag, tokenizer)
            except Exception:
                pass
        ids = tokenizer(frag, add_special_tokens=False, truncation=False).input_ids
        for t in ids:
            if len(tokens) == CHUNK:
                # break at the last comma within the final 20 tokens so words aren't split across chunks
                comma = tokenizer.convert_tokens_to_ids(",</w>")
                cut = max((i for i, x in enumerate(tokens) if x == comma and i >= CHUNK - 20), default=-1)
                if cut >= 0:
                    carry_t, carry_w = tokens[cut + 1:], weights[cut + 1:]
                    tokens, weights = tokens[:cut + 1], weights[:cut + 1]
                    close()
                    tokens, weights = carry_t, carry_w
                else:
                    close()
            tokens.append(t)
            weights.append(w)
    if tokens or not chunks:
        close()
    return chunks


def _encode_chunk(te, ids, ws, device, layer, final_norm, torch):
    """layer: hidden state index from the end (1 = last). Returns (embeds [1,77,D], pooled or None)."""
    ids_t = torch.tensor([ids], device=device)
    out = te(ids_t, output_hidden_states=True)
    if layer <= 1 and final_norm is None:
        z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    else:
        z = out.hidden_states[-layer]
        if final_norm is not None:
            z = final_norm(z)
    pooled = getattr(out, "text_embeds", None)
    w = torch.tensor(ws, device=z.device, dtype=z.dtype).view(1, -1, 1)
    if not torch.all(w == 1):
        orig_mean = z.float().mean()
        z = z * w
        new_mean = z.float().mean()
        if new_mean != 0:
            z = z * (orig_mean / new_mean).to(z.dtype)
    return z, pooled


def _final_norm(te):
    tm = getattr(te, "text_model", None)
    if tm is not None and hasattr(tm, "final_layer_norm"):
        return tm.final_layer_norm
    return getattr(te, "final_layer_norm", None)


def _pad_to(chunks, n, empty):
    return chunks + [empty] * (n - len(chunks))


def encode(pipe, family, prompt, negative, clip_skip=1, cfg=7.0):
    """Returns the keyword arguments to pass to the pipeline instead of prompt/negative_prompt.
    clip_skip uses A1111 numbering (1 = off; SDXL's normal reading is 2)."""
    import torch
    device = pipe.text_encoder.device if pipe.text_encoder is not None else pipe.text_encoder_2.device
    dtype = (pipe.unet.dtype if getattr(pipe, "unet", None) is not None else torch.float16)
    with torch.no_grad():
        if family == "sd15":
            tok, te = pipe.tokenizer, pipe.text_encoder
            pos = tokenize_chunks(pipe, tok, prompt)
            neg = tokenize_chunks(pipe, tok, negative)
            n = max(len(pos), len(neg))
            empty = tokenize_chunks(pipe, tok, "")[0]
            pos, neg = _pad_to(pos, n, empty), _pad_to(neg, n, empty)
            norm = _final_norm(te) if clip_skip > 1 else None
            layer = clip_skip if clip_skip > 1 else 1
            pe = torch.cat([_encode_chunk(te, i, w, device, layer, norm, torch)[0] for i, w in pos], dim=1)
            ne = torch.cat([_encode_chunk(te, i, w, device, layer, norm, torch)[0] for i, w in neg], dim=1)
            return {"prompt_embeds": pe.to(dtype), "negative_prompt_embeds": ne.to(dtype)}

        # SDXL: two text encoders, second-to-last layer of each (plus any extra CLIP skip), joined
        extra = max(0, clip_skip - 2)
        tok1, te1, tok2, te2 = pipe.tokenizer, pipe.text_encoder, pipe.tokenizer_2, pipe.text_encoder_2

        def both(text):
            c1 = tokenize_chunks(pipe, tok1, text)
            c2 = tokenize_chunks(pipe, tok2, text)
            n = max(len(c1), len(c2))
            return _pad_to(c1, n, tokenize_chunks(pipe, tok1, "")[0]), _pad_to(c2, n, tokenize_chunks(pipe, tok2, "")[0])

        p1, p2 = both(prompt)
        n1, n2 = both(negative)
        n = max(len(p1), len(n1))
        e1, e2 = tokenize_chunks(pipe, tok1, "")[0], tokenize_chunks(pipe, tok2, "")[0]
        p1, p2, n1, n2 = _pad_to(p1, n, e1), _pad_to(p2, n, e2), _pad_to(n1, n, e1), _pad_to(n2, n, e2)

        def run(c1, c2):
            embeds, pooled = [], None
            for (i1, w1), (i2, w2) in zip(c1, c2):
                a, _ = _encode_chunk(te1, i1, w1, te1.device if te1 is not None else device, 2 + extra, None, torch)
                b, pb = _encode_chunk(te2, i2, w2, te2.device, 2 + extra, None, torch)
                embeds.append(torch.cat([a.to(b.dtype), b], dim=-1))
                if pooled is None:
                    pooled = pb
            return torch.cat(embeds, dim=1), pooled

        pe, pp = run(p1, p2)
        if not (negative or "").strip() and getattr(pipe.config, "force_zeros_for_empty_prompt", False):
            ne, npool = torch.zeros_like(pe), torch.zeros_like(pp)
        else:
            ne, npool = run(n1, n2)
        return {"prompt_embeds": pe.to(dtype), "negative_prompt_embeds": ne.to(dtype),
                "pooled_prompt_embeds": pp.to(dtype), "negative_pooled_prompt_embeds": npool.to(dtype)}
