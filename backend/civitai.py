"""Civitai: add models and LoRAs from civitai.com (Settings → Models → From Civitai).

Paste a link to a model page (https://civitai.com/models/12345?modelVersionId=67890), a download link
(https://civitai.com/api/download/models/67890) or a number. The app reads the model's details from Civitai's
public API, then downloads the file straight into the right folder:
  * LoRAs (LORA / LoCon / DoRA): the LoRA folder set to its type (SD 1.5 / SDXL / FLUX / ERNIE) — one of yours if
    you have one, else Protoart's own;
  * SD 1.5 / SDXL checkpoints (incl. Pony / Illustrious / NoobAI): your checkpoints folder;
  * SD 1.5 / SDXL embeddings (Civitai calls them "Textual Inversion"): Protoart's embeddings folder.
Trigger words and details go next to the file as <name>.civitai.info (read like any other LoRA's).
Some downloads need a free Civitai account: paste an API key in Settings (kept on this Mac, sent only to Civitai).
Pure Python (urllib) so it's unit-testable; network only when you ask for it.
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://civitai.com/api/v1"
UA = "ProtoartStudio/1.1 (+https://civitai.com)"

# Civitai "baseModel" -> (family, flavour)
_BASES = [
    (r"^sd ?1\.[45]|^sd 1|^sd1", ("sd15", None)),
    (r"pony", ("sdxl", "pony")),
    (r"illustrious", ("sdxl", "illustrious")),
    (r"noob", ("sdxl", "noobai")),
    (r"^sdxl", ("sdxl", None)),
    (r"^flux[ .-]?2", ("flux2", None)),
    (r"^flux", ("flux", None)),
    (r"ernie", ("ernie", None)),
]
LORA_TYPES = ("LORA", "LoCon", "DoRA", "LyCORIS")


def family_of(base_model):
    b = (base_model or "").strip().lower()
    for pat, res in _BASES:
        if re.search(pat, b):
            return res
    return (None, None)


def parse_ref(text):
    """-> (model_id, version_id); either may be None."""
    t = (text or "").strip()
    if not t:
        raise ValueError("Paste a Civitai link (or a model number).")
    m = re.search(r"api/download/models/(\d+)", t)
    if m:
        return None, int(m.group(1))
    m = re.search(r"modelVersionId=(\d+)", t)
    v = int(m.group(1)) if m else None
    m = re.search(r"civitai\.(?:com|green|red)/models/(\d+)", t)
    if m:
        return int(m.group(1)), v
    if t.isdigit():
        return int(t), None
    raise ValueError("That doesn't look like a Civitai link. Copy the address of the model's page "
                     "(https://civitai.com/models/…).")


def _urlopen(req, timeout):
    """urllib.request.urlopen, but when a firewall/VPN blocks Python's connection ('[Errno 9] Bad file
    descriptor'), try once more through macOS's curl (net.use_curl_for_urllib)."""
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, OSError) as e:
        import net
        if not net.is_blocked_connection(e) or not net.use_curl_for_urllib():
            raise
        return urllib.request.urlopen(req, timeout=timeout)


def _reach_error(e):
    reason = getattr(e, "reason", e)
    if any(t in str(reason).lower() for t in ("bad file descriptor", "connection abort", "curl couldn't connect")):
        return ValueError("Couldn't reach Civitai: the connection was blocked on this Mac. If you use a firewall "
                          "(Little Snitch, LuLu…) or a VPN, allow Protoart Studio (and curl) to connect to "
                          "civitai.com, then press Retry.")
    return ValueError(f"Couldn't reach Civitai ({reason}). Check your internet connection, then press Retry.")


def _get_json(url, token=None, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with _urlopen(req, timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError("Civitai doesn't have that model (or it was removed).")
        if e.code in (401, 403):
            raise ValueError("Civitai wants you to log in for this one: add your Civitai API key in Settings → Models.")
        raise ValueError(f"Civitai answered with an error ({e.code}). Try again in a moment.")
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as e:
        raise _reach_error(e)


def _pick_file(files):
    """The model file itself: the primary .safetensors if there is one (never a pickled .pt/.ckpt when a
    safetensors exists)."""
    model_files = [f for f in files or [] if (f.get("type") or "Model") in ("Model", "Pruned Model")]
    if not model_files:
        return None
    safe = [f for f in model_files if (f.get("metadata") or {}).get("format") == "SafeTensor"
            or str(f.get("name", "")).lower().endswith(".safetensors")]
    pool = safe or model_files
    return next((f for f in pool if f.get("primary")), pool[0])


def allowed_extensions(kind):
    # embeddings: .pt too (most older ones only come as .pt); they're read with torch's weights-only loader,
    # which can't run code. Everything else: .safetensors only.
    return (".safetensors", ".pt") if kind == "embedding" else (".safetensors",)


def describe(model, version):
    """Turn Civitai's API answer into what the app needs. model: /models/{id} JSON (may be partial),
    version: one entry of model['modelVersions'] or /model-versions/{id} JSON."""
    mtype = (model.get("type") or (version.get("model") or {}).get("type") or "").strip()
    fam, flavor = family_of(version.get("baseModel"))
    f = _pick_file(version.get("files"))
    kind = ("lora" if mtype in LORA_TYPES else "checkpoint" if mtype == "Checkpoint"
            else "embedding" if mtype == "TextualInversion" else "vae" if mtype.upper() == "VAE"
            else "controlnet" if mtype.lower() == "controlnet" else (mtype.lower() or "unknown"))
    notes, ok = [], True
    if f is None:
        ok = False
        notes.append("This version has no model file to download.")
    elif not str(f.get("name", "")).lower().endswith(allowed_extensions(kind)):
        ok = False
        notes.append("Only .safetensors files are downloaded (other formats can run hidden code)."
                     if kind != "embedding" else "Only .safetensors and .pt embeddings are downloaded.")
    if kind == "checkpoint" and fam not in ("sd15", "sdxl"):
        ok = False
        notes.append(f"{version.get('baseModel') or 'This'} checkpoints can't be loaded from a single file yet: "
                     "SD 1.5, SDXL, Pony, Illustrious and NoobAI ones can.")
    elif kind == "embedding" and fam not in ("sd15", "sdxl"):
        ok = False
        notes.append(f"{version.get('baseModel') or 'This'} embeddings can't be used: SD 1.5 and SDXL ones "
                     "(Pony, Illustrious and NoobAI included) can.")
    elif kind == "vae" and fam not in ("sd15", "sdxl"):
        ok = False
        notes.append(f"{version.get('baseModel') or 'This'} VAEs can't be used: SD 1.5 and SDXL ones "
                     "(Pony, Illustrious and NoobAI included) can.")
    elif kind == "controlnet" and fam not in ("sd15", "sdxl"):
        ok = False
        notes.append(f"{version.get('baseModel') or 'This'} ControlNets can't be used: SD 1.5 and SDXL ones "
                     "(Pony, Illustrious and NoobAI included) can.")
    elif kind == "controlnet" and fam == "sdxl":
        notes.append("SDXL ControlNets are big (often 1–2.5 GB) and slow on 8 GB Macs. Control-LoRA, LLLite and Union "
                     "files aren't supported yet: the app checks the file after downloading and tells you.")
    elif kind == "lora" and fam is None:
        notes.append(f"Its base model ({version.get('baseModel') or 'not given'}) isn't one this app knows; "
                     "it's saved in your LoRA folder and its type is read from the file.")
    elif kind not in ("lora", "checkpoint", "embedding", "vae", "controlnet"):
        ok = False
        notes.append(f"{mtype or 'This kind of file'} isn't supported yet "
                     "(LoRAs, checkpoints, embeddings, VAEs and SD 1.5 / SDXL ControlNets are).")
    name = (model.get("name") or (version.get("model") or {}).get("name") or "civitai model").strip()
    # a checkpoint whose page offers its own VAE: downloaded with it and used with it automatically
    vf = next((x for x in version.get("files") or [] if (x.get("type") or "").upper() == "VAE"
               and str(x.get("name", "")).lower().endswith(".safetensors")), None) if kind == "checkpoint" else None
    return {
        "model_id": model.get("id") or version.get("modelId"), "version_id": version.get("id"),
        "name": name, "version": version.get("name") or "", "type": mtype, "kind": kind,
        "base_model": version.get("baseModel") or "", "family": fam, "flavor": flavor,
        "nsfw": bool(model.get("nsfw") or (version.get("model") or {}).get("nsfw")),
        "trained_words": [w for w in (version.get("trainedWords") or []) if isinstance(w, str)][:40],
        "file": f and {"name": f.get("name"), "size_mb": round(float(f.get("sizeKB") or 0) / 1024, 1),
                       "url": f.get("downloadUrl") or f"https://civitai.com/api/download/models/{version.get('id')}"},
        "vae_file": vf and {"name": vf.get("name"), "size_mb": round(float(vf.get("sizeKB") or 0) / 1024, 1),
                            "url": vf.get("downloadUrl")},
        "ok": ok, "notes": notes,
        "page": f"https://civitai.com/models/{model.get('id') or version.get('modelId')}",
    }


def inspect(text, token=None, get=_get_json):
    model_id, version_id = parse_ref(text)
    if version_id and not model_id:
        version = get(f"{API}/model-versions/{version_id}", token)
        return describe(version.get("model") or {}, version)
    model = get(f"{API}/models/{model_id}", token)
    versions = model.get("modelVersions") or []
    if not versions:
        raise ValueError("That model has no versions to download.")
    version = next((v for v in versions if v.get("id") == version_id), versions[0])
    return describe(model, version)


def find_duplicates(info, dirs, max_files=20000, limit=5):
    """Files you may already have: a download of the same Civitai version (its .civitai.info sidecar), a file
    with the same name, or a model file of exactly the same size. Searches dirs (and folders inside them)."""
    f = info.get("file") or {}
    name = str(f.get("name") or "").lower()
    size = int(round(float(f.get("size_mb") or 0) * 1024 * 1024))
    vid = info.get("version_id")
    exts = allowed_extensions(info.get("kind")) + (".ckpt", ".bin")
    found, seen, n = [], set(), 0
    for root in dirs:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in files:
                n += 1
                if n > max_files or len(found) >= limit:
                    return found
                p = os.path.join(dirpath, fn)
                low = fn.lower()
                why = None
                if low.endswith(".civitai.info") and vid:
                    try:
                        with open(p) as fh:
                            if json.load(fh).get("modelVersionId") == vid:
                                why = "the same Civitai version, downloaded before"
                                base = p[: -len(".civitai.info")]
                                p = next((base + e for e in exts if os.path.exists(base + e)), p)
                    except (OSError, ValueError, AttributeError):
                        pass
                elif name and low == name:
                    why = "a file with the same name"
                elif size > 1024 * 1024 and low.endswith(exts):
                    try:
                        if abs(os.path.getsize(p) - size) <= max(2048, size // 5000):
                            why = "a file of the same size"
                    except OSError:
                        pass
                if why and os.path.realpath(p) not in seen:
                    seen.add(os.path.realpath(p))
                    found.append({"path": p, "why": why})
    return found


def safe_filename(name, exts=(".safetensors",)):
    name = os.path.basename(str(name or "model.safetensors")).replace("\x00", "")
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .") or "model.safetensors"
    return name if name.lower().endswith(tuple(exts)) else name + exts[0]


def download_vae(info, dest_dir, token=None, progress=None, cancelled=lambda: False):
    """The VAE that came with a checkpoint (its Civitai page lists one): into the VAE folder, once.
    Returns the saved path, or None when there isn't one or it's already there."""
    vf = info.get("vae_file") or {}
    if not vf.get("url") or not vf.get("name"):
        return None
    if os.path.exists(os.path.join(dest_dir, safe_filename(vf["name"], (".safetensors",)))):
        return None
    sub = dict(info, file=vf, kind="vae", ok=True, type="VAE", vae_file=None, trained_words=[])
    return download(sub, dest_dir, token, progress, cancelled)


def download(info, dest_dir, token=None, progress=None, cancelled=lambda: False, stall_s=120):
    """Download info['file'] into dest_dir (resumes a .part left by an earlier try). Never overwrites a file.
    Writes <name>.civitai.info beside it. Returns the saved path."""
    f = info.get("file") or {}
    if not info.get("ok") or not f.get("url"):
        raise ValueError(" ".join(info.get("notes") or ["Nothing to download."]))
    os.makedirs(dest_dir, exist_ok=True)
    final = os.path.join(dest_dir, safe_filename(f.get("name"), allowed_extensions(info.get("kind"))))
    if os.path.exists(final):
        raise ValueError(f"{os.path.basename(final)} is already in {dest_dir}.")
    part = final + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    url = f["url"]
    if token:  # as a parameter: Civitai redirects to a storage server that rejects extra Authorization headers
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"token": token})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if token:
        # also as a header, for civitai.com only: "unredirected", so the file server it redirects to never gets it
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = _urlopen(req, 60)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            if token:
                raise ValueError(f"Civitai turned down your API key ({e.code}). It may be mistyped, "
                                 "deleted or expired: make a new one on civitai.com (Account settings → API Keys) and "
                                 "paste it here. Some models also need you to open their page on civitai.com once "
                                 "(Early Access or a licence to accept) before they can be downloaded.")
            raise ValueError("Civitai only lets signed-in accounts download this one. Paste your "
                             "free Civitai API key below (civitai.com → Account settings → API Keys → Add API key), "
                             "then press Download again.")
        if e.code == 416:  # already complete
            resp = None
        else:
            raise ValueError(f"Civitai answered with an error ({e.code}).")
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as e:
        raise _reach_error(e)
    if resp is not None:
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype or "application/json" in ctype:
            resp.close()  # a login page or an error message instead of the file
            raise ValueError("Civitai sent a page instead of the file: this download probably needs your "
                             "Civitai API key (Settings → Models).")
        if have and resp.status != 206:
            have = 0  # the server restarted from the beginning
        total = int(resp.headers.get("Content-Length") or 0) + have
        last = time.time()
        with resp, open(part, "ab" if have else "wb") as out:
            got = have
            while True:
                if cancelled():
                    raise InterruptedError()
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                now = time.time()
                if progress and now - last > 0.5:
                    progress(got, total)
                    last = now
        if total and got < total:
            raise ValueError("The download stopped before the end; press Download again to carry on.")
    os.replace(part, final)
    side = {"source": "civitai", "modelId": info.get("model_id"), "modelVersionId": info.get("version_id"),
            "name": info.get("name"), "version": info.get("version"), "baseModel": info.get("base_model"),
            "type": info.get("type"), "nsfw": info.get("nsfw"), "trainedWords": info.get("trained_words") or [],
            "page": info.get("page")}
    if info.get("vae_file"):
        side["vae"] = os.path.splitext(safe_filename(info["vae_file"].get("name"), (".safetensors",)))[0]
    try:
        with open(os.path.splitext(final)[0] + ".civitai.info", "w") as fh:
            json.dump(side, fh, indent=1)
    except OSError:
        pass
    return final
