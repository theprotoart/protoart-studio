"""✨ Enhance with Z-Engineer (BennyDaBall, Apache-2.0): 4B prompt-writing models fine-tuned from Z-Image-Turbo's own
text encoder (Qwen3 4B), V4 and V6. They're published only as full-size files and GGUF, so a one-time set-up on this
Mac downloads the full-size model (~8 GB, temporary), shrinks it to a 4-bit MLX version (~2.3 GB, kept with the
downloaded models) and deletes the full-size copy. It runs in the picture reader's environment (mlx-vlm, which brings mlx-lm).
"""
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import paths
import vlm

HERE = Path(__file__).resolve().parent

MODELS = {
    "zeng_v6": {"label": "Z-Engineer V6", "repo": "BennyDaBall/Z-Image-Engineer-V6",
                "page": "https://huggingface.co/BennyDaBall/Z-Image-Engineer-V6"},
    "zeng_v4": {"label": "Z-Engineer V4", "repo": "BennyDaBall/Qwen3-4b-Z-Image-Engineer-V4",
                "page": "https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Engineer-V4",
                "note": "built on an uncensored (abliterated) base: it writes whatever it's asked for"},
}
FULL_GB, KEEP_GB = 8.1, 2.4

# Our own wording of the author's recommended brief (one paragraph, staging, light, optics, no meta talk)
SYSTEM = ("You are a prompt engineer for text-to-image models. Treat the user's idea as production intent and expand "
          "it into one polished paragraph of about 150 to 220 words. Keep the subject, action, setting, mood and every "
          "explicit detail exactly as given. Add concrete staging in the foreground, midground and background, "
          "plausible lighting with its direction and colour, textures and materials, and lens, framing and depth of "
          "field (keep any the user gave). Never repeat a phrase. No camera brand names, no bullet points, no line "
          "breaks, no meta commentary. Output the paragraph only, in English.")


class ZengError(Exception):
    pass


def folder(mid):
    """Kept with the downloaded models (Settings → Model folders → Downloaded models folder), so it goes to an
    external drive when that folder is on one; a copy made before, in the data folder, is still found."""
    if mid not in MODELS:  # only the known helper models: an id is never a path (security audit, path traversal)
        raise ValueError("Unknown prompt helper model.")
    old = paths.HOME / "llm" / f"{mid}-mlx-4bit"
    if (old / "config.json").exists():
        return old
    return paths.hub_cache() / "protoart-mlx" / f"{mid}-mlx-4bit"


def ready(mid):
    d = folder(mid)
    return mid in MODELS and (d / "config.json").exists() and any(d.glob("*.safetensors")) and vlm.installed()


def status():
    return [{"id": k, "label": v["label"], "ready": ready(k), "page": v["page"], "note": v.get("note", ""),
             "full_gb": FULL_GB, "keep_gb": KEEP_GB} for k, v in MODELS.items()]


def _run_py(code, args, emit, cancel_event, what, timeout=None):
    """Run a short script in the picture reader's Python, passing JSON lines back."""
    p = subprocess.Popen([str(vlm.PY), "-c", code, json.dumps(args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, cwd=str(HERE), env=dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="0"))
    tail = []

    def read_err():
        for line in p.stderr:
            tail.append(line.rstrip())
            del tail[:-30]
            print(f"[{what}]", line.rstrip(), file=sys.stderr, flush=True)

    threading.Thread(target=read_err, daemon=True).start()
    result = None
    for line in p.stdout:
        if cancel_event is not None and cancel_event.is_set():
            p.kill()
            from engine import Cancelled
            raise Cancelled()
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("type") == "status":
            emit({"type": "status", "msg": msg["msg"]})
        elif msg.get("type") in ("done", "error"):
            result = msg
    p.wait()
    if not result or result.get("type") == "error":
        raise ZengError((result or {}).get("msg") or "\n".join(tail[-6:]) or f"{what} stopped")
    return result


_CONVERT = r'''
import json, os, shutil, sys
a = json.loads(sys.argv[1])
def say(m): print(json.dumps({"type": "status", "msg": m}), flush=True)
try:
    from huggingface_hub import snapshot_download
    say("Z-Engineer: downloading the full-size model (~8 GB, temporary)…")
    src = snapshot_download(a["repo"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    say("Z-Engineer: shrinking it to a 4-bit Mac version (a few minutes)…")
    from mlx_lm import convert
    out = a["out"]
    shutil.rmtree(out, ignore_errors=True)
    convert(hf_path=src, mlx_path=out, quantize=True, q_bits=4, q_group_size=64)
    if not any(f.endswith(".safetensors") for f in os.listdir(out)):
        raise RuntimeError("the shrunk model wasn't written")
    print(json.dumps({"type": "done", "src": src}), flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(json.dumps({"type": "error", "msg": f"{type(e).__name__}: {e}"}), flush=True)
'''

_WRITE = r'''
import json, sys
a = json.loads(sys.argv[1])
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    model, tok = load(a["path"])
    msgs = [{"role": "system", "content": a["system"]}, {"role": "user", "content": a["prompt"]}]
    try:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    text = generate(model, tok, prompt=prompt, max_tokens=int(a.get("max_tokens") or 420), verbose=False,
                    sampler=make_sampler(temp=float(a.get("temperature") or 0.8), top_p=0.92),
                    logits_processors=make_logits_processors(repetition_penalty=1.1))
    print(json.dumps({"type": "done", "text": text}), flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(json.dumps({"type": "error", "msg": f"{type(e).__name__}: {e}"}), flush=True)
'''


def setup(mid, emit, cancel_event=None):
    """One-time: picture reader's environment (if needed) -> mlx-lm -> download, shrink, delete the full-size copy."""
    if mid not in MODELS:
        raise ZengError("That isn't one of the Z-Engineer models.")
    if not vlm.supported():
        raise ZengError("Z-Engineer needs a Mac with Apple Silicon (M1 or newer).")
    if ready(mid):
        return {"id": mid, "ready": True}
    out = folder(mid)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free / 2**30  # both the download and the kept copy go there
    if free < FULL_GB + KEEP_GB + 1:
        raise ZengError(f"Setting up {MODELS[mid]['label']} needs about {FULL_GB + KEEP_GB + 1:.0f} GB free for a "
                        f"while (the full-size model is deleted afterwards); only {free:.1f} GB is free.")
    vlm.install(emit, cancel_event)  # the picture reader's environment (mlx-vlm), if it isn't there yet
    chk = subprocess.run([str(vlm.PY), "-c", "import mlx_lm"], capture_output=True)
    if chk.returncode != 0:
        emit({"type": "status", "msg": "Z-Engineer: installing the text model runner (one time)…"})
        vlm._run([str(vlm.PY), "-m", "pip", "install", "mlx-lm"], emit, cancel_event, "Installing mlx-lm")
    r = _run_py(_CONVERT, {"repo": MODELS[mid]["repo"], "out": str(out)}, emit, cancel_event, "zeng-setup")
    src = r.get("src") or ""
    # the full-size download was only needed for shrinking: delete it (only the app's own download, in the HF cache)
    if src and "models--" in src:
        repo_dir = Path(src).parents[1]
        if repo_dir.name.startswith("models--") and "BennyDaBall" in repo_dir.name:
            shutil.rmtree(repo_dir, ignore_errors=True)
    emit({"type": "status", "msg": f"{MODELS[mid]['label']} is ready for ✨ Enhance"})
    return {"id": mid, "ready": ready(mid)}


def write(mid, idea, shape, angle=None, emit=None, cancel_event=None, temperature=0.8):
    if not ready(mid):
        raise ZengError(f"{MODELS.get(mid, {}).get('label', mid)} isn't set up")
    extra = f" This time, {angle}." if angle else ""
    r = _run_py(_WRITE, {"path": str(folder(mid)), "system": SYSTEM, "prompt": f"{idea}\n(The image is {shape}.{extra})",
                         "temperature": temperature}, emit or (lambda m: None), cancel_event, "zeng")
    text = str(r.get("text") or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return " ".join(text.replace("<think>", "").split()).strip()


def remove(mid):
    shutil.rmtree(folder(mid), ignore_errors=True)
