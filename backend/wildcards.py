"""Wildcards: a different prompt for each image.

  {red|blue|green} dress      picks one option per image (options can be nested: {a {big|small}|b})
  {2$$red|blue|green}          picks 2 different options, joined with ", "
  __hair__                     picks a random line from ~/.protoartstudio/wildcards/hair.txt
                               (sub-folders work too: __clothes/tops__ -> wildcards/clothes/tops.txt)

The choice comes from the image's own seed, so the same seed gives the same prompt again.
"""
import random
import re

import paths

_FILE = re.compile(r"__([\w\-./ ]+?)__")
MAX_DEPTH = 8


def folder():
    return paths.HOME / "wildcards"


def has(text):
    return bool(text) and ("{" in text or bool(_FILE.search(text)))


def list_files():
    d = folder()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.rglob("*.txt")):
        name = p.relative_to(d).with_suffix("").as_posix()
        try:
            n = sum(1 for ln in p.read_text(errors="ignore").splitlines() if ln.strip() and not ln.startswith("#"))
        except OSError:
            n = 0
        out.append({"name": name, "token": f"__{name}__", "count": n})
    return out


def _lines(name):
    p = folder() / (name.strip() + ".txt")
    try:
        p.resolve().relative_to(folder().resolve())  # stay inside the wildcards folder
    except (ValueError, OSError):
        return []
    try:
        return [ln.strip() for ln in p.read_text(errors="ignore").splitlines() if ln.strip() and not ln.startswith("#")]
    except OSError:
        return []


def _innermost_braces(text):
    """Position of an innermost {...} group, or None."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            start = i
        elif ch == "}" and start is not None:
            return start, i
    return None


def _split_options(body):
    return body.split("|")


def resolve(text, rng):
    """Replace every wildcard in text using rng (a random.Random)."""
    if not has(text):
        return text
    for _ in range(MAX_DEPTH):
        changed = False
        # files first (their lines may contain {..} groups or other files)
        def file_sub(m):
            lines = _lines(m.group(1))
            return rng.choice(lines) if lines else m.group(0)
        new = _FILE.sub(file_sub, text)
        changed |= new != text
        text = new
        # then innermost {a|b} groups, repeatedly
        guard = 0
        while guard < 500:
            pos = _innermost_braces(text)
            if pos is None:
                break
            a, b = pos
            body = text[a + 1:b]
            count = 1
            m = re.match(r"^(\d+)\$\$(.*)$", body, re.S)
            if m:
                count, body = int(m.group(1)), m.group(2)
            opts = _split_options(body)
            if count <= 1:
                pick = rng.choice(opts)
            else:
                pick = ", ".join(rng.sample(opts, min(count, len(opts))))
            text = text[:a] + pick + text[b + 1:]
            changed = True
            guard += 1
        if not changed or not has(text):
            break
    return re.sub(r"\s{2,}", " ", text).strip()


def for_image(text, seed):
    """The prompt for one image: same seed -> same choices."""
    return resolve(text, random.Random(f"wild:{seed}:{text}"))
