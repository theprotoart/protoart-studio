"""Image -> prompt.

Two modes:
  * "describe": Florence-2-large writes a natural-language description (best for ERNIE / SDXL base).
  * "tags":     the WD v3 tagger (SmilingWolf) outputs Danbooru-style tags (best for anime SDXL/SD1.5 models).
"""
import csv

import numpy as np

FLORENCE_REPO = "florence-community/Florence-2-large"
WD_REPO = "SmilingWolf/wd-vit-tagger-v3"
WD_SIZE = 448
# Tag categories in selected_tags.csv
CAT_RATING, CAT_GENERAL, CAT_CHARACTER = 9, 0, 4
def _ram_gb():
    import os
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0


DETAIL_TASKS = {"short": "<CAPTION>", "detailed": "<DETAILED_CAPTION>", "very_detailed": "<MORE_DETAILED_CAPTION>"}


class Interrogator:
    def __init__(self, device, emit=None):
        self.device = device
        self.emit = emit or (lambda m: None)
        self._florence = None
        self._wd = None

    # ------------------------------------------------------------ Florence-2
    def describe(self, image, detail="detailed"):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        if self._florence is None:
            self.emit({"type": "status", "msg": "Loading Florence-2 (first time downloads ~1.5 GB)…"})
            dtype = torch.float16 if self.device == "mps" else torch.float32
            model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_REPO, torch_dtype=dtype).to(self.device).eval()
            proc = AutoProcessor.from_pretrained(FLORENCE_REPO)
            self._florence = (model, proc, dtype)
        model, proc, dtype = self._florence
        task = DETAIL_TASKS.get(detail, "<DETAILED_CAPTION>")
        self.emit({"type": "status", "msg": "Describing image…"})
        inputs = proc(text=task, images=image.convert("RGB"), return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            # beam search triples memory/time; use plain greedy decoding on smaller Macs
            beams = 3 if _ram_gb() >= 16 else 1
            ids = model.generate(**inputs, max_new_tokens=200, num_beams=beams)
        text = proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(text, task=task, image_size=image.size)
        caption = parsed.get(task, text) if isinstance(parsed, dict) else str(parsed)
        return " ".join(str(caption).replace("<pad>", "").split())

    # ------------------------------------------------------------ WD tagger
    def _load_wd(self):
        if self._wd is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            self.emit({"type": "status", "msg": "Loading Danbooru tagger (first time downloads ~380 MB)…"})
            from huggingface_hub import try_to_load_from_cache

            def get(name):
                hit = try_to_load_from_cache(WD_REPO, name)  # local copy first: no internet once downloaded
                return hit if isinstance(hit, str) else hf_hub_download(WD_REPO, name)
            model_path = get("model.onnx")
            tags_path = get("selected_tags.csv")
            with open(tags_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            names = [r["name"] for r in rows]
            cats = np.array([int(r["category"]) for r in rows])
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._wd = (sess, names, cats)
        return self._wd

    def tags(self, image, general_threshold=0.35, character_threshold=0.85, include_characters=True):
        from PIL import Image

        sess, names, cats = self._load_wd()
        self.emit({"type": "status", "msg": "Tagging image…"})
        img = image.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        canvas.alpha_composite(img)
        img = canvas.convert("RGB")
        side = max(img.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        square = square.resize((WD_SIZE, WD_SIZE), Image.BICUBIC)
        arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR, values 0-255
        arr = np.ascontiguousarray(arr[None, ...])
        inp = sess.get_inputs()[0].name
        probs = sess.run(None, {inp: arr})[0][0]

        picked = []
        for i, p in enumerate(probs):
            c = cats[i]
            if (c == CAT_GENERAL and p >= general_threshold) or \
               (include_characters and c == CAT_CHARACTER and p >= character_threshold):
                picked.append((float(p), c, names[i]))
        # characters first, then general tags by confidence
        picked.sort(key=lambda t: (t[1] != CAT_CHARACTER, -t[0]))
        rating_idx = [i for i, c in enumerate(cats) if c == CAT_RATING]
        rating = names[max(rating_idx, key=lambda i: probs[i])] if rating_idx else None
        return [format_tag(n) for _, _, n in picked], rating


def format_tag(name):
    """Danbooru tag -> prompt text: underscores become spaces (short emoticon tags like ^_^ stay as-is)."""
    return name.replace("_", " ") if len(name) > 3 else name
