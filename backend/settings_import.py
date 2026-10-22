"""Read the settings a picture was made with (or a .json of settings) so Create can load them.

Understands:
  * A1111 / Forge / SD.Next / Fooocus-style PNG "parameters" text (also in JPEG / WebP EXIF user comments)
  * ComfyUI: the "prompt" (API) graph in a PNG, or a ComfyUI .json (API format or a saved workflow)
  * Protoart Studio's own settings (PNG "protoart" text, or a .json exported from the app)
  * InvokeAI ("invokeai_metadata") and DiffusionBee-style plain JSON with prompt / seed / steps keys
Everything is read only: nothing in the file is run.
"""
import json
import re
from pathlib import Path

FIELDS = ("prompt", "negative_prompt", "steps", "cfg", "sampler", "seed", "width", "height", "model_name", "loras",
          "clip_skip", "strength", "source")


def _num(v, cast=float):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ A1111 text
_LORA_TAG = re.compile(r"<lora:([^:>]+):([0-9.]+)>")


def parse_a1111(text):
    """'prompt\\nNegative prompt: …\\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x768, Model: x'"""
    text = (text or "").strip()
    if not text:
        return None
    lines = text.split("\n")
    params_line = ""
    if lines and re.match(r"^\s*Steps:\s*\d+", lines[-1]):
        params_line = lines.pop()
    prompt_lines, neg_lines, in_neg = [], [], False
    for ln in lines:
        if ln.startswith("Negative prompt:"):
            in_neg = True
            ln = ln[len("Negative prompt:"):].strip()
        (neg_lines if in_neg else prompt_lines).append(ln)
    out = {"prompt": "\n".join(prompt_lines).strip(), "negative_prompt": "\n".join(neg_lines).strip(), "source": "A1111"}
    params = {}
    for m in re.finditer(r'\s*([\w \-/]+):\s*("(?:\\.|[^"\\])*"|[^,]*)(?:,|$)', params_line):
        params[m.group(1).strip().lower()] = m.group(2).strip().strip('"')
    out["steps"] = _num(params.get("steps"), int)
    out["cfg"] = _num(params.get("cfg scale"))
    out["seed"] = _num(params.get("seed"), int)
    out["sampler"] = params.get("sampler")
    if params.get("schedule type"):
        out["scheduler"] = params["schedule type"]
    if params.get("size") and "x" in params["size"]:
        w, h = params["size"].lower().split("x", 1)
        out["width"], out["height"] = _num(w, int), _num(h, int)
    out["model_name"] = params.get("model")
    out["clip_skip"] = _num(params.get("clip skip"), int)
    out["strength"] = _num(params.get("denoising strength"))
    loras = [{"name": n, "weight": float(w)} for n, w in _LORA_TAG.findall(out["prompt"])]
    if loras:
        out["loras"] = loras
        out["prompt"] = _LORA_TAG.sub("", out["prompt"]).replace(" ,", ",").strip(" ,")
    return out


# ------------------------------------------------------------------ ComfyUI
def parse_comfy(graph):
    """ComfyUI API graph {"3": {"class_type": "KSampler", "inputs": {...}}, …} (or a saved workflow)."""
    if isinstance(graph, dict) and "nodes" in graph and isinstance(graph["nodes"], list):
        return _parse_comfy_workflow(graph)
    if not isinstance(graph, dict):
        return None
    nodes = {k: v for k, v in graph.items() if isinstance(v, dict) and "class_type" in v}
    if not nodes:
        return None

    def node(ref):
        return nodes.get(str(ref[0])) if isinstance(ref, list) and ref else None

    def text_of(ref):
        n = node(ref)
        if not n:
            return ""
        inp = n.get("inputs", {})
        for key in ("text", "text_g", "prompt", "string"):
            if isinstance(inp.get(key), str):
                return inp[key]
            if isinstance(inp.get(key), list):
                return text_of(inp[key])
        return ""

    out = {"source": "ComfyUI", "loras": []}
    sampler = next((n for n in nodes.values() if "KSampler" in n["class_type"] or n["class_type"].startswith("SamplerCustom")), None)
    if sampler:
        inp = sampler.get("inputs", {})
        out["seed"] = _num(inp.get("seed", inp.get("noise_seed")), int)
        out["steps"] = _num(inp.get("steps"), int)
        out["cfg"] = _num(inp.get("cfg"))
        out["sampler"] = inp.get("sampler_name")
        out["scheduler"] = inp.get("scheduler")
        out["strength"] = _num(inp.get("denoise"))
        out["prompt"] = text_of(inp.get("positive"))
        out["negative_prompt"] = text_of(inp.get("negative"))
    for n in nodes.values():
        ct, inp = n["class_type"], n.get("inputs", {})
        if ct in ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader") and not out.get("model_name"):
            out["model_name"] = inp.get("ckpt_name") or inp.get("unet_name")
        elif "LoraLoader" in ct and inp.get("lora_name"):
            out["loras"].append({"name": Path(str(inp["lora_name"])).stem, "weight": _num(inp.get("strength_model")) or 1.0})
        elif ct in ("EmptyLatentImage", "EmptySD3LatentImage") and not out.get("width"):
            out["width"], out["height"] = _num(inp.get("width"), int), _num(inp.get("height"), int)
        elif ct == "CLIPSetLastLayer" and inp.get("stop_at_clip_layer") is not None:
            out["clip_skip"] = abs(_num(inp.get("stop_at_clip_layer"), int) or 1)
    if not out.get("prompt"):  # no sampler hooked up the usual way: take the longest text box
        texts = sorted((n.get("inputs", {}).get("text") for n in nodes.values()
                        if isinstance(n.get("inputs", {}).get("text"), str)), key=len, reverse=True)
        if texts:
            out["prompt"] = texts[0]
    return out if (out.get("prompt") or out.get("seed") is not None) else None


def _parse_comfy_workflow(wf):
    """A saved ComfyUI workflow (the UI format): widgets_values instead of named inputs."""
    out = {"source": "ComfyUI workflow", "loras": []}
    texts = []
    for n in wf.get("nodes", []):
        t, wv = n.get("type", ""), n.get("widgets_values") or []
        if t.startswith("KSampler") and len(wv) >= 6:
            # [seed, control_after_generate, steps, cfg, sampler, scheduler, denoise]
            out["seed"], out["steps"], out["cfg"] = _num(wv[0], int), _num(wv[2], int), _num(wv[3])
            out["sampler"], out["scheduler"] = wv[4], wv[5]
            if len(wv) > 6:
                out["strength"] = _num(wv[6])
        elif t == "CLIPTextEncode" and wv and isinstance(wv[0], str):
            texts.append((n.get("title") or "", wv[0]))
        elif t.startswith("CheckpointLoader") and wv:
            out["model_name"] = wv[0]
        elif "LoraLoader" in t and wv:
            out["loras"].append({"name": Path(str(wv[0])).stem, "weight": _num(wv[1]) if len(wv) > 1 else 1.0})
        elif t == "EmptyLatentImage" and len(wv) >= 2:
            out["width"], out["height"] = _num(wv[0], int), _num(wv[1], int)
    neg = [x for title, x in texts if "neg" in title.lower()]
    pos = [x for title, x in texts if "neg" not in title.lower()]
    if pos:
        out["prompt"] = pos[0]
    if neg:
        out["negative_prompt"] = neg[0]
    elif len(pos) > 1:
        out["negative_prompt"] = pos[1]
    return out if (out.get("prompt") or out.get("seed") is not None) else None


# ------------------------------------------------------------------ plain JSON (Protoart, InvokeAI, others)
_KEYS = {"prompt": ("prompt", "positive_prompt", "positive"), "negative_prompt": ("negative_prompt", "negative", "negativePrompt"),
         "steps": ("steps", "num_inference_steps"), "cfg": ("cfg", "cfg_scale", "guidance_scale", "scale"),
         "seed": ("seed",), "width": ("width",), "height": ("height",), "sampler": ("sampler", "sampler_name", "scheduler"),
         "model_name": ("model", "model_name", "ckpt_name"), "clip_skip": ("clip_skip",), "strength": ("strength", "denoise")}


def parse_plain(d):
    if not isinstance(d, dict):
        return None
    if isinstance(d.get("settings"), dict):  # a Protoart picture record
        d = {**d, **d["settings"]}
    out = {"source": "settings"}
    for field, keys in _KEYS.items():
        for k in keys:
            v = d.get(k)
            if isinstance(v, dict):  # InvokeAI {"model": {"name": …}}
                v = v.get("name") or v.get("model_name")
            if v not in (None, ""):
                out[field] = v
                break
    if isinstance(d.get("loras"), list):
        out["loras"] = [{"name": str(l.get("name") or l.get("model", {}).get("name") or ""), "weight": _num(l.get("weight")) or 1.0}
                        for l in d["loras"] if isinstance(l, dict)]
    for f, c in (("steps", int), ("seed", int), ("width", int), ("height", int), ("cfg", float), ("clip_skip", int), ("strength", float)):
        if f in out:
            out[f] = _num(out[f], c)
    return out if isinstance(out.get("prompt"), str) and out["prompt"].strip() else None


def from_json_text(text):
    try:
        d = json.loads(text)
    except Exception:
        return None
    if isinstance(d, dict) and d.get("app") == "Protoart Studio" and d.get("kind") == "settings":
        d = d.get("settings") or {}
    return parse_comfy(d) or parse_plain(d) or (parse_plain(d.get("invokeai_metadata")) if isinstance(d, dict) else None)


# ------------------------------------------------------------------ files
def read(path):
    """-> dict of the FIELDS found (missing ones left out), or raises ValueError."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        if p.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("That .json file is too big to be settings.")
        res = from_json_text(p.read_text(encoding="utf-8", errors="replace"))
        if not res:
            raise ValueError("No picture settings found in that .json (expected a ComfyUI workflow, A1111 / InvokeAI "
                             "settings, or one exported from Protoart Studio).")
        return _clean(res)
    from PIL import Image
    with Image.open(p) as im:
        info = dict(im.info or {})
        exif = im.getexif() if hasattr(im, "getexif") else {}
    for key in ("beestudio", "protoart", "Protoart", "settings"):
        if isinstance(info.get(key), str):
            r = from_json_text(info[key])
            if r:
                return _clean(dict(r, source="Protoart Studio"))
    if isinstance(info.get("parameters"), str):
        r = parse_a1111(info["parameters"])
        if r:
            return _clean(r)
    for key in ("prompt", "workflow"):
        if isinstance(info.get(key), str):
            r = from_json_text(info[key])
            if r:
                return _clean(r)
    if isinstance(info.get("invokeai_metadata"), str):
        r = from_json_text(info["invokeai_metadata"])
        if r:
            return _clean(dict(r, source="InvokeAI"))
    comment = exif.get(0x9286) if exif else None  # JPEG / WebP: EXIF UserComment (A1111 writes it there)
    if comment is not None:
        raw = comment if isinstance(comment, bytes) else str(comment).encode("utf-8", "ignore")
        txt = raw[8:].decode("utf-16-be", "ignore") if raw[:8] == b"UNICODE\x00" else raw.decode("utf-8", "ignore")
        r = parse_a1111(txt.strip("\x00 ")) or from_json_text(txt)
        if r:
            return _clean(r)
    for v in info.values():  # any other text chunk holding JSON or A1111 text
        if isinstance(v, str) and len(v) > 10:
            r = from_json_text(v) or (parse_a1111(v) if "Steps:" in v else None)
            if r:
                return _clean(r)
    raise ValueError("No settings found in that picture (it may have been saved without them, or by an app that "
                     "doesn't write them).")


def _clean(d):
    out = {k: v for k, v in d.items() if v not in (None, "", [])}
    for k in ("prompt", "negative_prompt"):
        if k in out:
            out[k] = str(out[k])[:8000]
    return out
