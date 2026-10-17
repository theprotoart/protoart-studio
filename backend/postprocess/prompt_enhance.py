"""Prompt enhancer: turns a short idea into a detailed prompt, in the style the chosen model understands.

  * Natural-language models (SDXL, ERNIE, newer ones): a vivid 2-4 sentence description.
  * SD 1.5: comma-separated tags (Danbooru-style where it fits) plus quality tags, which is what
    SD 1.5 models respond to best.

Uses a small instruction-tuned language model (Qwen2.5, Apache 2.0) so it runs in seconds even on
an 8 GB Mac: 0.5B (~1 GB) on Macs under 16 GB, 1.5B (~3 GB) on bigger ones. (The previous enhancer,
ERNIE's own 3B prompt model in 4-bit, took minutes per prompt on small Macs and looked frozen.)
Downloads once, then works offline.
"""
import re

SMALL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
LARGE_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
DOWNLOAD_GB = {SMALL_REPO: 1.0, LARGE_REPO: 3.1}

SYSTEM = {
    "natural": (
        "You expand short image ideas into one vivid, detailed description for a text-to-image model. "
        "Describe the subject, its appearance, the setting, lighting, colours, composition, mood and art style in "
        "2 to 4 sentences. Keep everything the user asked for and don't change the subject. "
        "No lists, no captions or text in the image, no explanations. Always write in English. "
        "Output only the description."),
    "tags": (
        "You turn short image ideas into a Stable Diffusion 1.5 prompt made of short comma-separated tags "
        "(Danbooru-style tags when they fit, e.g. 1girl, long hair, smile). Start with the user's subject, then add "
        "15 to 25 tags for details, clothing, pose, expression, background, lighting, composition and art style. "
        "No sentences. Always write in English. Output only the tags on one line."),
    "edit": (
        "You rewrite a short request for changing a picture into one clear editing instruction for an image editing "
        "model. Say exactly what to change (and how it should look afterwards) and that everything else stays the "
        "same. One or two sentences, under 45 words. Don't describe the whole picture. Always write in English. "
        "Output only the instruction."),
}
EXAMPLES = {
    "natural": ("a fox in snow",
                "A red fox with a thick winter coat stands alert in a snowy pine forest at dawn, its breath "
                "visible in the cold air. Soft golden light filters through frosted branches, casting long blue "
                "shadows across the fresh snow. Shallow depth of field, crisp detail in the fur, a calm and quiet mood, "
                "in the style of a wildlife photograph."),
    "tags": ("a girl with a sword",
             "1girl, solo, holding sword, katana, long hair, black hair, red eyes, determined expression, "
             "fighting stance, school uniform, pleated skirt, wind, falling leaves, autumn, sunset, dramatic lighting, "
             "dynamic angle, cowboy shot, detailed background, anime style"),
    "edit": ("make her jacket red",
             "Change the woman's jacket to a deep red leather jacket with the same cut and fit, keeping her face, "
             "pose, hair, the background and the lighting exactly as they are."),
}
QUALITY_TAGS = ["masterpiece", "best quality", "highly detailed"]


def style_for(family):
    """Tags for the Stable Diffusion models (SD 1.5, SDXL incl. Pony / Illustrious / NoobAI), sentences for the
    models that read language (ERNIE, FLUX, FLUX.2); "edit" rewrites an editing instruction (Edit with words)."""
    if family == "edit":
        return "edit"
    return "tags" if family in ("sd15", "sdxl") else "natural"


class PromptEnhancer:
    def __init__(self, device, emit=None, ram_gb=8):
        self.device = device
        self.emit = emit or (lambda m: None)
        self.repo = LARGE_REPO if ram_gb >= 16 else SMALL_REPO
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.emit({"type": "status", "msg": "Loading prompt enhancer…"})
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.repo)
            self._model = AutoModelForCausalLM.from_pretrained(self.repo, torch_dtype=dtype).to(self.device).eval()
        return self._tok, self._model

    def enhance(self, prompt, width=1024, height=1024, cancel_event=None, family="sdxl"):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        style = style_for(family)
        tok, model = self._load()
        shape = "portrait" if height > width * 1.1 else "landscape" if width > height * 1.1 else "square"
        ex_in, ex_out = EXAMPLES[style]
        messages = [
            {"role": "system", "content": SYSTEM[style]},
            {"role": "user", "content": ex_in},
            {"role": "assistant", "content": ex_out},
            {"role": "user", "content": f"{prompt}\n(The image is {shape}. Answer in English.)"},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(self.device)
        start = inputs["input_ids"].shape[1]
        emit = self.emit
        limit = {"tags": 110, "edit": 80}.get(style, 170)

        class Watch(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                n = input_ids.shape[1] - start
                if n and n % 15 == 0:
                    emit({"type": "status", "msg": f"Enhancing prompt… ({n} words)"})
                    emit({"type": "progress", "image": 1, "images": 1, "step": min(n, limit), "steps": limit})
                return bool(cancel_event is not None and cancel_event.is_set())

        emit({"type": "status", "msg": "Enhancing prompt…"})
        # Small models drift into Chinese: block non-Latin vocabulary with a precomputed mask.
        # (transformers' suppress_tokens compares every token against every blocked id, which asked the Mac's
        # GPU for a 7.7 GB buffer and crashed the engine; a mask needs ~150 KB.)
        mask = self._block_mask(model, scores_device=self.device)

        class BlockForeign(LogitsProcessor):
            def __call__(self, input_ids, scores):
                m = mask if mask.shape[0] == scores.shape[-1] else None
                if m is None:
                    return scores
                return scores.masked_fill(m.unsqueeze(0), float("-inf"))

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=limit, do_sample=True, temperature=0.6, top_p=0.9,
                                 repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
                                 logits_processor=LogitsProcessorList([BlockForeign()]),
                                 stopping_criteria=StoppingCriteriaList([Watch()]))
        result = tok.decode(out[0][start:], skip_special_tokens=True)
        return clean(result, prompt, style)

    def _block_mask(self, model, scores_device):
        if getattr(self, "_mask", None) is None:
            import torch
            vocab = model.get_output_embeddings().weight.shape[0]
            m = torch.zeros(vocab, dtype=torch.bool)
            ids = [i for i in self._non_english_tokens() if i < vocab]
            if ids:
                m[torch.tensor(ids, dtype=torch.long)] = True
            self._mask = m.to(scores_device)
        return self._mask

    def _non_english_tokens(self):
        """Ids of vocabulary pieces containing non-Latin script (Chinese, Japanese, Korean, Cyrillic…).
        Blocking them keeps the answer in English. Computed once (~1 s)."""
        if getattr(self, "_blocked", None) is None:
            tok = self._tok
            blocked = []
            for i in range(len(tok)):
                piece = tok.decode([i])
                if any(_foreign(ch) for ch in piece):
                    blocked.append(i)
            self._blocked = blocked
        return self._blocked

    def unload(self):
        self._model = None
        self._tok = None
        self._mask = None


def _foreign(ch):
    o = ord(ch)
    if o < 0x250:            # ASCII + accented Latin
        return False
    if 0x2000 <= o <= 0x206F or 0x20A0 <= o <= 0x20CF or 0x2190 <= o <= 0x21FF:  # punctuation, currency, arrows
        return False
    if o == 0xFFFD:          # partial byte pieces decode to this; they're needed for normal text too
        return False
    return True


def clean(text, original, style):
    """Tidy the model's answer; fall back to the original prompt if it's unusable."""
    t = "".join(ch for ch in (text or "") if not _foreign(ch))  # belt and braces: drop any non-Latin text
    t = " ".join(t.replace("\n", " ").split())
    t = re.sub(r"^(prompt|description|tags|output)\s*:\s*", "", t, flags=re.IGNORECASE).strip().strip('"').strip()
    if len(t) < 8:
        return original
    if style == "tags":
        seen, tags = set(), []
        for raw in re.split(r",|\.|;", t):
            tag = raw.strip().strip("*-").strip()
            if tag and len(tag) <= 40 and tag.lower() not in seen:
                seen.add(tag.lower())
                tags.append(tag)
        # keep the user's own words first, quality tags at the end
        orig = [o.strip() for o in original.split(",") if o.strip()]
        head = [o for o in orig if o.lower() not in seen]
        tags = head + tags
        for q in QUALITY_TAGS:
            if q not in {x.lower() for x in tags}:
                tags.append(q)
        # SD 1.5 reads only ~75 tokens of a prompt; anything past that is cut off, so keep the list short.
        # Room for the quality tags is kept first; details fill the rest in order.
        tok = lambda t: len(re.findall(r"\w+|[^\w\s]", t)) + 1  # rough token count, +1 for the comma
        quality = [t for t in tags if t.lower() in QUALITY_TAGS]
        budget = 70 - sum(tok(q) for q in quality)
        kept = []
        for tag in tags:
            if tag.lower() in QUALITY_TAGS:
                continue
            if tok(tag) > budget:
                break
            kept.append(tag)
            budget -= tok(tag)
        kept += quality
        return ", ".join(kept)
    return t
