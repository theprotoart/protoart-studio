"""MiniMax-H3-Image (MiniMax, 33B; MiniMax-H3-Image Community License) on Apple Silicon, also on 8 GB Macs.

How it fits: Comfy-Org's pruned INT8 ConvRot files (49 GB, pinned) are downloaded to the downloaded-models folder, and a
4-bit copy (~27 GB) is made once on this Mac's own drive (h3_worker.convert). Each picture runs in four short processes
(prompt reader, reference pictures, picture model, decoder) with h3-image-mlx (vendor/, Apache-2.0); the picture model
streams from the drive one block at a time, so a picture needs under 2 GB of memory. Measured on an 8 GB M1:
1024x1024 in about 4 minutes.

Licence: the model may only be used in its Applicable Territory (worldwide except the EU, the UK, the Republic of
Korea and the United States), commercial products must show its name, and its pictures may not be used to improve
other AI models. Nothing here downloads or runs it before the person has accepted that (accept_licence)."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
WORKER = HERE / "h3_worker.py"
MANIFEST = json.loads((HERE / "vendor" / "h3_image_mlx" / "weights.json").read_text())
FILES = MANIFEST["files"]  # dit, text encoder, vae, tokenizer, vae config (in that order)
DOWNLOAD_GB = sum(f["bytes"] for f in FILES) / 2**30  # ~45.9 GiB (49.3 GB)
FOUR_GB = 27.0  # the 4-bit copy on this Mac's drive
LICENCE_NAME = "MiniMax-H3-Image Community License"
LICENCE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE"
EXCLUDED = "the United States, the European Union, the United Kingdom and the Republic of Korea"
SIZES = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768),
         (1536, 1536), (2048, 2048), (1536, 2048), (2048, 1536)]


class H3Error(Exception):
    pass


class Cancelled(Exception):
    pass


def downloads():
    return paths.hub_cache() / "h3-image"


def four():
    return paths.HOME / "h3-image-4bit"


def files():
    dit, text, vae, tok, cfg = (downloads() / f["path"] for f in FILES)
    return {"dit_int8": dit, "text_int8": text, "vae": vae, "tokenizer": tok, "vae_config": cfg,
            "dit": four() / "dit-4bit.safetensors", "text": four() / "text-4bit.safetensors"}


# ------------------------------------------------------------------------------------------------------ licence
def _licence_file():
    return paths.HOME / "h3-image-licence.json"


def licence_accepted():
    try:
        return json.loads(_licence_file().read_text()).get("accepted") is True
    except (OSError, ValueError):
        return False


def accept_licence(accepted=True):
    _licence_file().write_text(json.dumps({"accepted": bool(accepted), "licence": LICENCE_NAME,
                                           "when": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return {"accepted": bool(accepted)}


# ------------------------------------------------------------------------------------------------------- state
def _have(entry):
    p = downloads() / entry["path"]
    try:
        return p.stat().st_size == entry["bytes"]
    except OSError:
        return False


def downloaded():
    return all(_have(f) for f in FILES)


def small_files_here():
    f = files()
    return all(f[k].exists() for k in ("vae", "tokenizer", "vae_config"))


def ready():
    """Everything a picture needs is on this Mac: the 4-bit copy and the small files."""
    f = files()
    return f["dit"].exists() and f["text"].exists() and small_files_here()


def status():
    f = files()
    need_dl = sum(e["bytes"] for e in FILES if not _have(e) and not (
        e is FILES[0] and f["dit"].exists() or e is FILES[1] and f["text"].exists())) / 2**30
    return {"accepted": licence_accepted(), "ready": ready(), "download_gb": round(need_dl, 1),
            "four_gb": 0 if f["dit"].exists() and f["text"].exists() else FOUR_GB,
            "downloads": str(downloads()), "four": str(four())}


def free_gb(p):
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


# ---------------------------------------------------------------------------------------------------- download
def _fetch(url, dest, size, on_bytes, cancel_event):
    """One file, resumable (a .part file is continued with a Range request)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    req = urllib.request.Request(url, headers={"User-Agent": "ProtoartStudio", **({"Range": f"bytes={done}-"} if done else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if done and r.status != 206:  # the server ignored the Range: start over
            done = 0
        with open(part, "ab" if done else "wb") as f:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                chunk = r.read(4 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                on_bytes(len(chunk))
    if part.stat().st_size != size:
        raise H3Error(f"{dest.name} came out the wrong size; press the button again to carry on.")
    part.replace(dest)


def download(on_event, cancel_event=None):
    """The 8-bit files (pinned revisions), from Hugging Face or, if that fails, the same files on ModelScope. Files the
    4-bit copy was already made from aren't needed again."""
    if not licence_accepted():
        raise H3Error(f"Accept the {LICENCE_NAME} first.")
    f = files()
    todo = [e for e in FILES if not _have(e) and not (e is FILES[0] and f["dit"].exists())
            and not (e is FILES[1] and f["text"].exists())]
    total = sum(e["bytes"] - _part_size(e) for e in todo)
    if not todo:
        return
    if free_gb(downloads()) < total / 2**30 + 2:
        raise H3Error(f"MiniMax-H3-Image needs about {total / 2**30:.0f} GB free in the downloaded-models folder "
                      f"({downloads().parent}); only {free_gb(downloads()):.0f} GB is free.")
    got = [0]
    last = [0.0]

    def tick(n):
        got[0] += n
        if time.time() - last[0] > 1.5:
            last[0] = time.time()
            on_event({"type": "download", "done": got[0], "total": total})

    for e in todo:
        dest = downloads() / e["path"]
        errors = []
        for src in ("huggingface", "modelscope"):
            try:
                _fetch(e[src], dest, e["bytes"], tick, cancel_event)
                errors = []
                break
            except Cancelled:
                raise
            except Exception as err:  # noqa: BLE001  (tried on the other site)
                errors.append(f"{src}: {err}")
        if errors:
            raise H3Error(f"Couldn't download {Path(e['path']).name} ({'; '.join(errors)[:300]}). "
                          "Check the internet connection and press the button again: it carries on where it stopped.")
    on_event({"type": "download", "done": total, "total": total})


def _part_size(e):
    try:
        return (downloads() / (e["path"] + ".part")).stat().st_size
    except OSError:
        return 0


# --------------------------------------------------------------------------------------------- worker processes
def memory():
    """(available RAM GB, swap used GB, total RAM GB), or None if it can't be read."""
    try:
        import psutil
        v = psutil.virtual_memory()
        return v.available / 2**30, psutil.swap_memory().used / 2**30, v.total / 2**30
    except Exception:
        return None


def starving(base, now):
    """The picture is pushing the Mac into heavy swapping: 3 GB more swap since the stage started and almost no RAM
    left. (A normal stage reads its weights from the file as it goes and needs under 2 GB, so it never gets here.)"""
    if not now:
        return False
    avail, swap, total = now
    return swap - base[1] >= 3 and avail < max(0.4, total * 0.05)


def _run(python, args, on_line, cancel_event=None):
    proc = subprocess.Popen([str(python), str(WORKER), *map(str, args)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=str(HERE))
    stop = threading.Event()
    starved = []
    base = memory()

    def watch():  # Stop ends the stage right away (its memory goes back to macOS)
        while not stop.is_set() and proc.poll() is None:
            if cancel_event is not None and cancel_event.wait(0.5):
                proc.kill()
                return
            if cancel_event is None:
                stop.wait(0.5)
            if base and starving(base, memory()):  # stop it before macOS swaps so hard the whole computer freezes
                starved.append(True)
                proc.kill()
                return

    threading.Thread(target=watch, daemon=True).start()
    err_tail = []

    def errs():
        for line in proc.stderr:
            err_tail.append(line)
            del err_tail[:-40]

    threading.Thread(target=errs, daemon=True).start()
    last = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                last = ev
                on_line(ev)
        proc.wait()
    finally:
        stop.set()
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    if starved:
        raise H3Error("MiniMax-H3-Image was stopped because this Mac ran out of memory (it would have frozen the "
                      "computer). Use a smaller size or fewer pictures, close other apps, and try again.")
    if proc.returncode:
        tail = "".join(err_tail)[-1500:]
        if "GPU Timeout" in tail or "kIOGPUCommandBufferCallbackErrorTimeout" in tail:
            raise H3Error("The graphics chip took too long (macOS stopped it). Close other apps and try again, or use a "
                          "smaller size.")
        print("[h3] worker failed:", tail, flush=True)
        raise H3Error(f"MiniMax-H3-Image stopped ({tail.strip().splitlines()[-1] if tail.strip() else proc.returncode}).")
    return last


def make_four(python, on_event, cancel_event=None):
    """The 4-bit copy on this Mac's own drive, made once from the 8-bit files (about 20-30 minutes from an external
    drive). The 8-bit files are kept: the app never deletes model files."""
    f = files()
    four().mkdir(parents=True, exist_ok=True)
    jobs = [(f["dit_int8"], f["dit"]), (f["text_int8"], f["text"])]
    need = sum(src.stat().st_size * 0.6 for src, dst in jobs if not dst.exists() and src.exists()) / 2**30
    if need and free_gb(four()) < need + 3:
        raise H3Error(f"Making MiniMax-H3-Image's 4-bit copy needs about {need:.0f} GB free on this Mac's drive; only "
                      f"{free_gb(four()):.0f} GB is free.")
    for n, (src, dst) in enumerate(jobs):
        if dst.exists():
            continue
        if not src.exists():
            raise H3Error(f"{src.name} isn't downloaded yet.")
        _run(python, ["convert", src, dst],
             lambda ev, n=n: ev.get("type") == "progress" and on_event({"type": "convert", "part": n + 1, "parts": 2,
                                                                         "done": ev["done"], "total": ev["total"]}),
             cancel_event)


def _cache_key(prompt, refs):
    h = hashlib.sha256()
    h.update(prompt.encode())
    for r in refs:
        h.update(hashlib.sha256(Path(r).read_bytes()).digest())
    try:
        st = files()["text"].stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        pass
    return h.hexdigest()[:32]


def picture(python, prompt, out, width=1024, height=1024, seed=42, refs=(), on_event=lambda ev: None,
            cancel_event=None, ref_sides=None, loras=None):
    """One picture -> out (PNG). refs: reference pictures (Edit: "<Picture 1>" is the first); ref_sides: the short side
    each goes in at (each costs as much work as the picture drawn: (side/32)² tokens). Returns the timings."""
    if not ready():
        raise H3Error("MiniMax-H3-Image isn't on this Mac yet.")
    if not (256 <= width <= 4096 and 256 <= height <= 4096 and width % 32 == 0 and height % 32 == 0):
        raise H3Error("MiniMax-H3-Image needs a width and height that are multiples of 32 (256 to 4096).")
    refs = [str(r) for r in refs][:16]
    cache_dir = paths.HOME / "h3-prompts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sides = list(ref_sides or [])[:len(refs)]
    cached = cache_dir / f"{_cache_key(prompt + json.dumps(sides), refs)}.safetensors"
    timings = {}
    with tempfile.TemporaryDirectory(prefix="h3-", dir=str(paths.HOME / "tmp") if (paths.HOME / "tmp").exists() else None) as tmp:
        work = Path(tmp)
        (work / "request.json").write_text(json.dumps({
            "prompt": prompt, "width": width, "height": height, "seed": int(seed) % 2**31, "refs": refs, "ref_sides": sides,
            "loras": [[str(f), float(w)] for f, w in (loras or [])],  # your MiniMax-H3 LoRAs: (file, strength)
            "paths": {k: str(v) for k, v in files().items()}}))
        stages = ["text", *(["refs"] if refs else []), "dit", "vae"]
        if cached.exists():
            shutil.copy(cached, work / "embedding.safetensors")
            stages.remove("text")
        for s in stages:
            on_event({"type": "stage", "stage": s})
            done = _run(python, ["stage", s, work], on_event, cancel_event)
            timings[s] = (done or {}).get("seconds")
            if s == "text":
                shutil.copy(work / "embedding.safetensors", cached)
        shutil.copy(work / "image.png", out)
    return timings


def edit_prompt(prompt, n_pictures):
    """H3 edits are written as "Edit <Picture 1>. …": add that when the prompt doesn't name its pictures."""
    p = (prompt or "").strip()
    if n_pictures and "<picture" not in p.lower():
        p = f"Edit <Picture 1>. {p}" if n_pictures == 1 else \
            f"Edit <Picture 1> using {', '.join(f'<Picture {i}>' for i in range(2, n_pictures + 1))}. {p}"
    return p
