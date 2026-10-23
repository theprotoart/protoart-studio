"""✨ Plan the layout (Create → Regions): one sentence → the regional prompter's areas.

Omost's idea (an LLM plans where things go, then regions draw them) in a size that fits an 8 GB Mac: the small prompt
enhancer model writes one line per area ("LEFT: a knight in armour"), and this module turns those lines into the
regional prompter's layout (columns, rows or a grid), area prompts and the scene prompt. When the model's answer
can't be used, the sentence's own position words ("on the left", "at the top"…) are read instead.
"""
import re

MAX_AREAS = 4

# position word -> (row, column); None = not said
_POS = {
    "left": (None, 0), "center": (None, 1), "centre": (None, 1), "right": (None, 2),
    "top": (0, None), "middle": (1, None), "bottom": (2, None),
    "top left": (0, 0), "top center": (0, 1), "top centre": (0, 1), "top right": (0, 2),
    "middle left": (1, 0), "middle right": (1, 2),
    "bottom left": (2, 0), "bottom center": (2, 1), "bottom centre": (2, 1), "bottom right": (2, 2),
    "upper left": (0, 0), "upper right": (0, 2), "lower left": (2, 0), "lower right": (2, 2),
    "foreground": (2, None), "background": (0, None), "sky": (0, None), "ground": (2, None),
}
_SCENE = ("scene", "setting", "background scene", "overall", "whole picture", "style")

SYSTEM = (
    "You plan the layout of a picture for an image model that draws each part from its own prompt. Read the user's "
    "description and answer with lines only, in this exact form:\n"
    "SCENE: the setting, lighting and style shared by the whole picture (no subjects)\n"
    "<PLACE>: one subject and how it looks\n"
    "PLACE is one of LEFT, CENTER, RIGHT, TOP, MIDDLE, BOTTOM, TOP LEFT, TOP RIGHT, BOTTOM LEFT, BOTTOM RIGHT. "
    "Use 2 to 4 places, one subject each, where the user put them (or where they fit best). Never put two subjects on "
    "one line: two cats are two lines (LEFT: the black cat, RIGHT: the white cat). Each line under 25 words. "
    "Always write in English. No other text.")
EXAMPLES = [
    ("a knight fighting a dragon under a stormy sky",
     "SCENE: a rocky battlefield under a dark stormy sky, dramatic lighting, epic fantasy painting\n"
     "LEFT: a knight in dented silver armour raising a longsword, red cape blowing in the wind\n"
     "RIGHT: a huge green dragon rearing up, wings spread, breathing fire"),
    ("a cat on a sofa with a window showing the moon above it",
     "SCENE: a cosy living room at night, warm lamp light, soft photographic style\n"
     "TOP: a window with a full moon in a deep blue night sky\n"
     "BOTTOM: a fluffy orange cat curled up on a green velvet sofa"),
]


def _clean(text):
    text = re.sub(r"\s+", " ", (text or "").strip(" \t-*•\"'"))
    return text.strip(" ,;.") if len(text) > 2 else ""


def parse_lines(raw):
    """The model's answer -> (scene, [(row, col, prompt)]). Unknown lines are ignored."""
    scene, items = "", []
    for line in (raw or "").splitlines():
        m = re.match(r"\s*[-*•\d.)\s]*\**\s*([A-Za-z ]{3,22}?)\s*\**\s*[:\-–]\s*(.+)$", line)
        if not m:
            continue
        key, text = m.group(1).strip().lower(), _clean(m.group(2))
        key = re.sub(r"^(the|in the|on the|at the)\s+", "", key)
        if not text:
            continue
        if key in _SCENE:
            scene = scene or text
        elif key in _POS and len(items) < MAX_AREAS:
            r, c = _POS[key]
            items.append((r, c, text))
    return scene, items


# "on the left", "to the right of the picture", "at the top", "in the sky above"… -> place words
_PHRASES = [
    (r"\b(?:in the |at the |on the )?(?:top|upper)[ -](left|right)\b", lambda m: "top " + m.group(1)),
    (r"\b(?:in the |at the |on the )?(?:bottom|lower)[ -](left|right)\b", lambda m: "bottom " + m.group(1)),
    (r"\b(?:on|to|at) the (left|right)(?: side)?\b", lambda m: m.group(1)),
    (r"\bin the (?:middle|center|centre)\b", lambda m: "center"),
    (r"\b(?:at the top|above|overhead|in the sky)\b", lambda m: "top"),
    (r"\b(?:at the bottom|below|underneath|in the foreground)\b", lambda m: "bottom"),
]


def _place_of(text):
    """(place word or None, the text without its position words)."""
    for pat, name in _PHRASES:
        m = re.search(pat, text, flags=re.I)
        if m:
            rest = text[:m.start()] + text[m.end():]
            return name(m).strip().lower(), re.sub(r"\bcorner\b", "", rest)
    return None, text


def from_words(sentence):
    """No model: split the sentence into parts at commas / 'and' / 'while' and read each part's position words.
    -> (scene, [(row, col, prompt)])."""
    parts = [p for p in re.split(r",|;|\band\b|\bwhile\b", sentence or "") if p.strip()]
    placed = [_place_of(p)[0] is not None for p in parts]
    if sum(placed) == 1:
        # only one place said, inside a "… with …" part ("a cabin in the snow with the northern lights above"):
        # the subject before "with" is the other area. (Not when several places are said: there "a man with an
        # umbrella in the middle" is one subject.)
        i = placed.index(True)
        m = re.search(r"\bwith\b", parts[i])
        if m and _place_of(parts[i][:m.start()])[0] is None and _place_of(parts[i][m.end():])[0] is not None:
            parts[i:i + 1] = [parts[i][:m.start()], parts[i][m.end():]]
    scene_bits, items = [], []
    for p in parts:
        place, p = _place_of(p)
        text = _clean(re.sub(r"\b(?:is|are|stands?|sits?)\s*$", "", p.strip()))
        if not text:
            continue
        if place in _POS and len(items) < MAX_AREAS:
            r, c = _POS[place]
            items.append((r, c, text))
        else:
            scene_bits.append(text)
    if len(items) == 1 and scene_bits:  # one subject placed: the first other subject takes the opposite side
        r, c, _t = items[0]
        if r in (0, 2):  # top / bottom (or a corner): the other row, full width
            items.append((2 - r, None, scene_bits.pop(0)))
        elif c in (0, 2):  # left / right
            items.append((None, 2 - c, scene_bits.pop(0)))
    return ", ".join(scene_bits), items


def _distinct(items):
    """A usable plan from the model: different subjects in different places (a small model sometimes puts the
    same subject in every area or repeats the whole sentence)."""
    seen, out = set(), []
    for r, c, t in items:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        out.append((r, c, t))
    common = {"the", "and", "with", "left", "right", "side", "top", "bottom", "middle", "center", "centre", "corner"}
    words = [set(re.findall(r"[a-z]{3,}", t.lower())) - common for _r, _c, t in out]
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a, b = words[i], words[j]
            if a and b and len(a & b) / min(len(a), len(b)) > 0.75:  # the same subject twice
                return []
    return out


def layout(items):
    """[(row, col, prompt)] -> {"layout", "ratios", "prompts"} for the regional prompter (prompts in its order)."""
    rows_said = {r for r, _c, _t in items if r is not None}
    cols_said = {c for _r, c, _t in items if c is not None}
    if not rows_said:  # side by side
        order = sorted(range(len(items)), key=lambda i: (items[i][1] if items[i][1] is not None else 1, i))
        return {"layout": "cols", "ratios": ",".join(["1"] * len(items)), "prompts": [items[i][2] for i in order]}
    if not cols_said:  # stacked
        order = sorted(range(len(items)), key=lambda i: (items[i][0] if items[i][0] is not None else 1, i))
        return {"layout": "rows", "ratios": ",".join(["1"] * len(items)), "prompts": [items[i][2] for i in order]}
    # a grid: rows top to bottom, each row's areas left to right (a place without a row goes in the middle row)
    grid = {}
    for i, (r, c, t) in enumerate(items):
        grid.setdefault(1 if r is None else r, []).append((1 if c is None else c, i, t, c is not None))
    rows = [sorted(grid[r]) for r in sorted(grid)]
    # keep the columns lined up ("a tree top left, a house bottom right, a lake bottom left"): a row whose places
    # were said gets an area for the rest of the scene (None) in the columns it doesn't use, if that stays ≤ 4 areas
    said = {c for row in rows for c, _i, _t, s in row if s}
    cols = sorted(said | {2 - c for c in said})  # "bottom right" alone still leaves room on the left
    filled = [row if not all(said for *_x, said in row) else
              sorted(row + [(c, 99, None, True) for c in cols if c not in {x[0] for x in row}]) for row in rows]
    if sum(len(r) for r in filled) <= MAX_AREAS:
        rows = filled
    rows = [[(c, i, t) for c, i, t, _s in row] for row in rows]
    if len(rows) == 1:  # every place was in one row after all
        return {"layout": "cols", "ratios": ",".join(["1"] * len(rows[0])), "prompts": [t for _c, _i, t in rows[0]]}
    return {"layout": "grid", "ratios": ";".join(",".join(["1"] * len(r)) for r in rows),
            "prompts": [t for r in rows for _c, _i, t in r]}


def plan(sentence, raw=None):
    """The plan for one sentence. The places you said yourself ("on the left…", "above…") come first; otherwise
    the model's answer `raw` (if its areas hold different subjects). Returns {"layout", "ratios", "prompts", "scene",
    "source"} or raises ValueError."""
    scene, items = from_words(sentence)
    source = "words"
    if len(items) < 2 and raw:
        scene, items = parse_lines(raw)
        items, source = _distinct(items), "model"
    if len(items) < 2:
        raise ValueError("Couldn't find separate parts in that description. Say where things go, e.g. “a knight on "
                         "the left, a dragon on the right, a stormy sky above”.")
    out = layout(items[:MAX_AREAS])
    out["prompts"] = [p if p is not None else (scene or "background scenery") for p in out["prompts"]]
    out.update(scene=scene or "", source=source)
    return out


def needs_model(sentence):
    """True when the sentence doesn't say where things go, so the model has to plan it."""
    return len(from_words(sentence)[1]) < 2
