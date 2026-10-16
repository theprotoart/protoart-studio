"""Content filter: decides what counts as NSFW (not safe for work) so the app can hide it.

Three sources, combined:
  * pictures: a small on-device image checker (Falconsai/nsfw_image_detection, Apache-2.0, ~340 MB,
    downloaded once when you turn it on) plus the words in the prompt;
  * models and LoRAs: words in their names and, for LoRAs, the tags they were trained on;
  * your own marks, which always win ("this is NSFW" / "this is fine").

Everything here runs on this Mac. The word checks are pure Python (unit-tested anywhere); the image checker
imports torch/transformers only when it's used.
"""
import json
import re
import threading

import paths

CHECKER_REPO = "Falconsai/nsfw_image_detection"
CHECKER_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
CHECKER_GB = 0.34
THRESHOLD = 0.5  # checker score at or above this = NSFW

# Words that mark a *prompt* as NSFW. Only the positive prompt is read: "nsfw" in a negative prompt means
# the opposite. Kept to clearly explicit terms so ordinary prompts ("chicken breast", "bare trees") pass.
PROMPT_WORDS = (
    "nsfw", "nude", "nudes", "nudity", "naked", "topless", "bottomless", "porn", "porno", "pornographic",
    "hentai", "explicit", "sex", "sexual", "intercourse", "erotic", "erotica", "lewd", "ecchi", "nipples",
    "nipple", "areola", "areolae", "genitals", "genitalia", "penis", "vagina", "vulva", "pussy", "cock",
    "cum", "orgasm", "fellatio", "blowjob", "handjob", "masturbation", "masturbating", "bdsm", "bondage",
    "xxx", "r18", "r-18", "uncensored", "rating_explicit", "rating:explicit", "rating_questionable",
    "bare breasts", "exposed breasts", "pubic hair", "spread legs",
)
# Words that mark a *model or LoRA name* as NSFW.
NAME_WORDS = ("nsfw", "nude", "nudes", "nudity", "naked", "porn", "porno", "hentai", "xxx", "lewd", "ecchi",
              "erotic", "erotica", "r18", "r-18", "uncensored", "sex", "sexy", "nipples", "pussy", "genitals",
              "bdsm", "fetish", "18+",
              # explicit acts and body words often used in LoRA names
              "fuck", "fucking", "fucked", "anal", "gape", "creampie", "cum", "cumshot", "bukkake", "ejaculation",
              "squirt", "squirting", "fellatio", "blowjob", "deepthroat", "handjob", "footjob", "paizuri",
              "masturbation", "orgasm", "ahegao", "futa", "futanari", "penis", "dick", "cock", "vagina", "vulva",
              "tits", "titty", "titties", "boobs", "breasts", "milf", "oral sex", "licking oral", "spread legs",
              "lingerie nude", "stripper", "onlyfans", "pov sex", "missionary", "doggystyle", "cowgirl sex")
# Training tags that mark a LoRA as NSFW when they're on a good share of its training pictures.
TAG_WORDS = ("nsfw", "nude", "nudity", "naked", "nipples", "pussy", "penis", "sex", "completely nude",
             "explicit", "rating_explicit", "rating:explicit", "uncensored", "topless", "areolae", "cum")
TAG_SHARE = 0.15


def _pattern(words):
    alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    # word boundaries that also treat _ and - as separators ("my_nsfw_lora", "nude-v2")
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])", re.I)


_PROMPT_RE = _pattern(PROMPT_WORDS)
# names: a version number may follow a word ("ejaculation1", "nude_v2")
_NAME_RE = re.compile(_pattern(NAME_WORDS).pattern.replace("(?![a-z0-9])", "(?![a-z])"), re.I)
# unambiguous words that are caught even inside run-together names ("hugefaketitsv2", "xxxcreampie")
NAME_INSIDE = ("creampie", "bukkake", "fellatio", "blowjob", "ejaculat", "deepthroat", "futanari", "ahegao",
               "paizuri", "hentai", "porn", "titties", "fakeits", "faketits", "bigtits", "hugetits", "nsfw",
               "pussy", "cumshot", "handjob", "footjob", "nipples", "masturbat")
_INSIDE_RE = re.compile("|".join(map(re.escape, NAME_INSIDE)), re.I)


def prompt_flag(prompt):
    """True if the (positive) prompt asks for NSFW content."""
    if not prompt:
        return False
    text = str(prompt).replace("_", " ") + " " + str(prompt)  # "bare_breasts" and "rating_explicit" both match
    return bool(_PROMPT_RE.search(text))


def name_flag(*names):
    """True if a model/LoRA name or file name says it's NSFW."""
    for n in names:
        if not n:
            continue
        n = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(n))  # camelCase: "realisticNSFW" -> "realistic NSFW"
        if _NAME_RE.search(n.replace("_", " ").replace(".", " ")) or _INSIDE_RE.search(n):
            return True
    return False


def meta_flag(meta):
    """True if a LoRA's training tags (kohya ss_tag_frequency) are largely explicit, or its metadata says so."""
    if not isinstance(meta, dict):
        return False
    for k in ("modelspec.nsfw", "nsfw", "civitai.nsfw"):
        if str(meta.get(k, "")).strip().lower() in ("true", "1", "yes"):
            return True
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return False
    try:
        freq = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return False
    try:
        n_images = int(meta.get("ss_num_train_images") or 0)
    except (TypeError, ValueError):
        n_images = 0
    tags = {}
    for bucket in (freq.values() if isinstance(freq, dict) else []):
        if isinstance(bucket, dict):
            for tag, count in bucket.items():
                t = str(tag).strip().lower()
                try:
                    tags[t] = tags.get(t, 0) + int(count)
                except (TypeError, ValueError):
                    pass
    if not tags:
        return False
    total = n_images or max(tags.values())
    explicit = max((c for t, c in tags.items() if t in TAG_WORDS), default=0)
    return explicit >= max(2, TAG_SHARE * total)


# ------------------------------------------------------------------ your own marks (models and LoRAs)
_lock = threading.Lock()


def _marks_file():
    return paths.HOME / "content_marks.json"


def load_marks():
    try:
        d = json.loads(_marks_file().read_text())
    except (OSError, ValueError):
        d = {}
    return {k: dict(d.get(k) or {}) for k in MARK_KINDS}


# "models"/"loras": your NSFW marks (True / False; missing = automatic).
# "people_models"/"people_loras": models and LoRAs you tagged as made of real people (True; missing = no).
MARK_KINDS = ("models", "loras", "people_models", "people_loras")


def set_mark(kind, name, value):
    """kind: one of MARK_KINDS. value: True, False or None (remove the mark)."""
    if kind not in MARK_KINDS:
        raise ValueError("unknown kind of mark")
    with _lock:
        d = load_marks()
        if value is None:
            d[kind].pop(name, None)
        else:
            d[kind][name] = bool(value)
        f = _marks_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(f)
        return d


def effective(auto, mark):
    """Your mark wins; otherwise the automatic guess."""
    return bool(auto) if mark is None else bool(mark)


# ------------------------------------------------------------------ the picture checker
def checker_installed():
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(CHECKER_REPO, f), str) for f in CHECKER_FILES)
    except Exception:
        return False


class Checker:
    """Scores pictures 0..1 (1 = NSFW). Runs on the CPU so it never competes with the image model for GPU
    memory; a picture takes a fraction of a second."""

    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
        path = snapshot_download(CHECKER_REPO, allow_patterns=CHECKER_FILES)  # offline unless allowed
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModelForImageClassification.from_pretrained(path).eval()
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.nsfw_index = next((i for i, v in labels.items() if v in ("nsfw", "porn", "sexy", "hentai")), 1)

    def score(self, image):
        image = image.convert("RGB")
        with self.torch.no_grad():
            logits = self.model(**self.proc(images=image, return_tensors="pt")).logits
            return round(float(logits.softmax(-1)[0, self.nsfw_index]), 4)
