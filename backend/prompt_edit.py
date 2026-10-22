"""Prompt editing, A1111 style: change part of the prompt part-way through the steps (SD 1.5 / SDXL).

  [cat:dog:0.5]   "cat" for the first half of the steps, then "dog"
  [cat:dog:12]    "cat" until step 12, then "dog"
  [hat:0.3]       nothing until 30 %, then "hat"   (adds it later: keeps the layout, changes the detail)
  [hat::0.3]      "hat" until 30 %, then nothing    (removes it later)

The rough layout of a picture is decided in the first steps and the detail in the last, so this mixes the
composition of one idea with the details of another. Square brackets without a ':' and a number are left alone
(they are the usual "less weight" brackets).
"""
import re

_EDIT = re.compile(r"\[([^\[\]:]*):([^\[\]:]*):\s*(\d+(?:\.\d+)?)\s*\]")   # [from:to:when]
_ADD = re.compile(r"\[([^\[\]:]+):\s*(\d+(?:\.\d+)?)\s*\]")               # [to:when]


def has_edits(prompt):
    return bool(_EDIT.search(prompt or "") or _ADD.search(prompt or ""))


def _fraction(when, steps):
    w = float(when)
    return w if w < 1 else min(1.0, w / max(1, steps))


def schedule(prompt, steps):
    """-> [(start fraction, prompt text), …] sorted, first one at 0.0. Plain prompts give one entry."""
    prompt = prompt or ""
    points = set()
    for m in _EDIT.finditer(prompt):
        points.add(_fraction(m.group(3), steps))
    for m in _ADD.finditer(prompt):
        points.add(_fraction(m.group(2), steps))
    starts = [0.0] + sorted(p for p in points if 0.0 < p < 1.0)

    def text_at(f):
        def edit(m):
            return m.group(1) if f < _fraction(m.group(3), steps) else m.group(2)

        def add(m):
            return "" if f < _fraction(m.group(2), steps) else m.group(1)

        t = _EDIT.sub(edit, prompt)
        t = _ADD.sub(add, t)
        t = re.sub(r"\s*,\s*(,\s*)+", ", ", t)  # tidy commas left by removed words
        return re.sub(r"\s{2,}", " ", t).strip(" ,")

    out = []
    for f in starts:
        txt = text_at(f)
        if not out or out[-1][1] != txt:
            out.append((f, txt))
    return out
