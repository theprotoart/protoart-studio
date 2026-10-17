"""Protoart Studio backend process.

Protocol (one JSON object per line):
  UI -> backend (stdin):   {"id": 7, "cmd": "generate", "args": {...}}
  backend -> UI (stdout):  {"type": "result", "id": 7, "data": ...}
                           {"type": "error",  "id": 7, "msg": "..."}
                           {"type": "status" | "progress" | "image" | "ready", ...}
Logs and library chatter go to stderr, so stdout only ever carries protocol lines.

Like DiffusionBee, the UI talks to Python over stdin/stdout; unlike DiffusionBee,
messages are structured JSON instead of 4-letter prefixes, and quick commands
(library, presets) are answered even while an image is generating.
"""
import json
import re
import os
import sys
import threading
import traceback
from pathlib import Path

# Keep stdout clean: protocol messages use a private copy of the real stdout,
# and anything else that prints (libraries, warnings) is redirected to stderr.
_proto_out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)


class _Tee:
    """Everything the engine prints also goes to ~/.protoartstudio/logs/backend.log, so the log survives
    the window reloading or the app closing (Settings -> Backend log shows the end of this file)."""

    def __init__(self, stream, path):
        self.stream = stream
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > 5_000_000:  # keep it small
                path.replace(path.with_suffix(".old.log"))
            self.file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        except OSError:
            self.file = None

    def write(self, s):
        self.stream.write(s)
        if self.file:
            try:
                self.file.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        self.stream.flush()
        if self.file:
            try:
                self.file.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


_home = os.environ.get("PROTOART_HOME") or os.environ.get("BEESTUDIO_HOME") or os.path.expanduser("~/.protoartstudio")
LOG_PATH = Path(_home) / "logs" / "backend.log"
sys.stderr = _Tee(sys.stderr, LOG_PATH)
sys.stdout = sys.stderr
import time as _t  # noqa: E402
print(f"\n===== Protoart Studio backend started {_t.strftime('%Y-%m-%d %H:%M:%S')} =====", file=sys.stderr, flush=True)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# The standalone app's own Python doesn't know where macOS keeps its certificates: point every library at
# certifi's list so the one-time downloads (models, the fast engine) work over HTTPS.
try:
    import certifi as _certifi
    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except Exception:
    pass
# Lets transformers load SDNQ-quantized text encoders (ERNIE 4/8-bit builds).
os.environ.setdefault("DIFFUSERS_SDNQ_TRANSFORMERS", "1")
try:
    _RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
except (ValueError, OSError, AttributeError):
    _RAM_GB = 0
if 0 < _RAM_GB <= 16:
    # Older/smaller Macs: let PyTorch's Metal backend spill to macOS's normal memory management
    # instead of stopping with "MPS backend out of memory" when it reaches its own cap.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402
paths.apply_hub_cache()  # the downloaded-models folder chosen in Settings, before any model library loads
import net  # noqa: E402  (puts the model libraries in offline/no-telemetry mode before they load)
import samplers  # noqa: E402
import loras  # noqa: E402
import nsfw as content  # noqa: E402
import civitai  # noqa: E402
import diffusionbee  # noqa: E402
import embeddings  # noqa: E402
import model_hub  # noqa: E402
from library import Library  # noqa: E402

_out_lock = threading.Lock()


def emit(msg):
    with _out_lock:
        _proto_out.write(json.dumps(msg) + "\n")
        _proto_out.flush()


def log(*a):
    print("[backend]", *a, file=sys.stderr, flush=True)


paths.ensure_dirs()
library = Library(paths.DB_PATH)
import thumbs  # noqa: E402
# After the rename from BeeStudio, fix image paths that still point into the old data folder.
if paths.HOME != paths.LEGACY_HOME and not paths.LEGACY_HOME.exists():
    library.rewrite_path_prefix(str(paths.LEGACY_HOME), str(paths.HOME))
diffusionbee.enabled = bool(library.get_setting("import_diffusionbee", True))
net.strict = bool(library.get_setting("strict_offline", False))
model_hub.LOCAL_MODELS[:] = [m for m in (library.get_setting("local_models", []) or []) if isinstance(m, dict)]
model_hub.MODEL_DIRS[:] = [d for d in (library.get_setting("model_dirs", []) or []) if isinstance(d, str)]
model_hub.MODEL_FAMILY.update({k: v for k, v in (library.get_setting("model_families", {}) or {}).items()
                               if isinstance(k, str) and v in ("sd15", "sdxl")})
model_hub.USER_MODELS[:] = [m for m in (library.get_setting("user_models", []) or []) if isinstance(m, dict)]
loras.EXTRA_DIRS[:] = [d for d in (library.get_setting("lora_dirs", []) or []) if isinstance(d, str)]
embeddings.EXTRA_DIRS[:] = [d for d in (library.get_setting("embedding_dirs", []) or []) if isinstance(d, str)]
net.apply_hf_token(library.get_setting("hf_token"))  # Hugging Face sign-in for gated models (if you saved one)
loras.FOLDER_FAMILY.update({k: v for k, v in (library.get_setting("lora_dir_families", {}) or {}).items()
                            if isinstance(k, str) and v in loras.FAMILIES})
loras.NAME_FAMILY.update({k: v for k, v in (library.get_setting("lora_families", {}) or {}).items()
                          if isinstance(k, str) and v in loras.FAMILIES})
_custom_images_dir = library.get_setting("images_dir")
if _custom_images_dir:
    paths.IMAGES_DIR = Path(_custom_images_dir)
    try:
        paths.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        log("custom image folder unavailable, using default:", _custom_images_dir)
        paths.IMAGES_DIR = paths.HOME / "images"
_engine = None
_engine_lock = threading.Lock()
PAUSED = threading.Event()  # Pause button: the running image waits at its next step, and no new job starts


def engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            if os.environ.get("BEESTUDIO_FAKE_ENGINE"):  # used by tests: no torch, no models
                from tests.fake_engine import FakeEngine as Engine
            else:
                from engine import Engine  # heavy import (torch) deferred until needed
            _engine = Engine(library, emit)
            _engine.pause_event = PAUSED
        return _engine


settings = {"force_load": False, "low_memory": False, "image_format": "png", "jpeg_quality": 92, "previews": True,
            "embed_meta": "details", "check_pictures": False, "sfw_mode": False}  # embed_meta: "details" (prompt + settings, no file paths) or "none"
HEAVY = ("heal", "edit_words", "vae_get", "vectorize", "preload", "nsfw_scan", "model_delete", "civitai_download", "downloads_move", "model_download", "generate", "postprocess", "interrogate", "enhance_prompt", "train_lora", "train_embedding", "lora_merge", "train_autocaption", "control_preview",
         "convert_tdict", "compare")


def _folder_bytes(root):
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def engine_cancelled():
    return _engine is not None and _engine.cancel_event.is_set()


def move_downloads(a, emit_progress):
    """Move every downloaded model from one folder to another (Settings -> Downloaded models -> Move)."""
    import shutil as _sh
    src, dst = Path(a["from"]).expanduser(), Path(a["to"]).expanduser()
    if not src.is_dir():
        return {"moved": 0}
    if src.resolve() == dst.resolve():
        return {"moved": 0}
    dst.mkdir(parents=True, exist_ok=True)
    entries = [e for e in src.iterdir() if e.name.startswith(("models--", "datasets--", "spaces--", ".locks"))]
    sizes = {e: _folder_bytes(e) for e in entries}
    total = max(1, sum(sizes.values()))
    done, moved, skipped = 0, 0, []
    for e in entries:
        if engine_cancelled():
            break
        target = dst / e.name
        emit_progress(done, total, f"Moving {e.name.replace('models--', '').replace('--', '/')}…")
        if target.exists():
            skipped.append(e.name)  # already there (e.g. moved before): leave both copies alone
        else:
            _sh.move(str(e), str(target))  # same drive: instant; another drive: copied, then the original removed
            moved += 1
        done += sizes[e]
    emit_progress(total, total, "Done")
    return {"moved": moved, "skipped": skipped, "gb": round(total / 1e9, 1)}


def job_label(cmd, a):
    if cmd == "compare":
        import grid
        p = ((a.get("base") or {}).get("prompt") or "").strip().replace("\n", " ")
        axes = grid.AXES.get(a.get("x_axis"), "?") + (" × " + grid.AXES.get(a.get("y_axis"), "?") if a.get("y_axis") else "")
        return f"Compare {axes} · " + (p[:60] + ("…" if len(p) > 60 else ""))
    if cmd == "downloads_move":
        return "Move downloaded models to " + str(a.get("to"))
    if cmd == "preload":
        import engine as _e
        return "Get " + str(_e.all_models().get(a.get("model"), {}).get("label", a.get("model"))) + " ready"
    if cmd == "nsfw_scan":
        return "Check Library pictures (content filter)"
    if cmd == "civitai_download":
        return "Download from Civitai · " + str(a.get("name") or a.get("url") or "")
    if cmd == "model_delete":
        import engine as _e
        return "Delete " + str(_e.all_models().get(a.get("model"), {}).get("label", a.get("model"))) + " from this Mac"
    if cmd == "model_download":
        import engine as _e
        spec = _e.all_models().get(a.get("model"), {})
        return f"Download {spec.get('label', a.get('model'))} (~{spec.get('download_gb', '?')} GB)"
    if cmd == "convert_tdict":
        n = len(a.get("files") or [])
        return f"Convert {n} DiffusionBee file{'s' if n != 1 else ''} to .safetensors"
    if cmd == "control_preview":
        return f"ControlNet map ({a.get('type')})"
    if cmd == "train_lora":
        return f"Train LoRA · {a.get('name') or 'my_lora'} · {a.get('steps') or '?'} steps"
    if cmd == "train_embedding":
        return f"Train embedding · {a.get('name') or 'my_embedding'} · {a.get('steps') or '?'} steps"
    if cmd == "lora_merge":
        return f"Merge {len(a.get('items') or [])} LoRAs → {a.get('name') or 'merged_lora'}"
    if cmd == "train_autocaption":
        return "Caption training images (" + ("tags" if a.get("mode") == "tags" else "describe") + ")"
    if cmd == "generate":
        n = int(a.get("num_images") or 1)
        p = (a.get("prompt") or "").strip().replace("\n", " ")
        return (f"{n}× " if n > 1 else "") + (p[:80] + ("…" if len(p) > 80 else "") or "(no prompt)")
    if cmd == "postprocess":
        return {"faces": "Restore faces", "enhance": "Enhance & upscale", "both": "Enhance + restore faces",
                "depth": "Depth map"}.get(a.get("action"), "Post-process") + \
            f" · image #{a.get('image_id')}"
    if cmd == "enhance_prompt":
        return "Enhance prompt"
    if cmd == "heal":
        return "Remove (spot heal)" + (f" · image #{a.get('image_id')}" if a.get("image_id") is not None else "")
    if cmd == "vae_get":
        return "Download the recommended VAE"
    if cmd == "edit_words":
        return "Edit with words · " + ((a.get("prompt") or "")[:40] or "FLUX.2")
    if cmd == "vectorize":
        return f"Vectorize · {a.get('colors') or 8} colours"
    return "Image → prompt (" + ("tags" if a.get("mode") == "tags" else "describe") + ")"


class JobQueue:
    """FIFO of heavy jobs. Any waiting job can be removed; the running one can be stopped.

    Every change is broadcast to the UI as {"type": "queue", "jobs": [...]}.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._waiting = []   # list of dicts: id, cmd, args, label, model
        self.running = None  # job dict currently executing
        self.closed = False

    def snapshot(self):
        jobs = ([dict(self._public(self.running), state="running")] if self.running else [])
        jobs += [dict(self._public(j), state="queued") for j in self._waiting]
        return jobs

    @staticmethod
    def _public(j):
        a = j.get("args") or {}
        info = {"id": j["id"], "cmd": j["cmd"], "label": j["label"], "model": j.get("model")}
        if j["cmd"] == "generate":  # what the UI needs to estimate how long the job will take
            info.update(steps=a.get("steps"), images=a.get("num_images") or 1, width=a.get("width"),
                        height=a.get("height"), strength=a.get("strength") if a.get("init_image") else None,
                        enhance=bool(a.get("enhance_prompt")), faces=bool(a.get("restore_faces")),
                        fast=bool(a.get("fast")), nsfw_prompt=content.prompt_flag(a.get("prompt")),
                        # which tab made it: the live preview shows only there (Create or Image to image)
                        i2i=bool(a.get("init_image") or (isinstance(a.get("control"), dict) and a["control"].get("type"))),
                        loras=[{"name": l.get("name"), "weight": l.get("weight", 1.0)} for l in (a.get("loras") or [])
                               if isinstance(l, dict)])
        elif j["cmd"] == "compare":
            b = a.get("base") or {}
            try:
                import grid
                cells = len(grid.parse_values(a.get("x_axis"), a.get("x_values"))) * \
                    (len(grid.parse_values(a.get("y_axis"), a.get("y_values"))) if a.get("y_axis") else 1)
            except Exception:
                cells = 1
            info.update(steps=b.get("steps"), images=cells, width=b.get("width"), height=b.get("height"),
                        model=b.get("model"), loras=[{"name": l.get("name"), "weight": l.get("weight", 1.0)}
                                                     for l in (b.get("loras") or []) if isinstance(l, dict)])
        elif j["cmd"] in ("train_lora", "train_embedding"):
            info.update(steps=a.get("steps"), images=1)
        elif j["cmd"] == "postprocess":
            info.update(action=a.get("action"), upscale=(a.get("options") or {}).get("upscale_factor"))
        return info

    def _broadcast(self):
        emit({"type": "queue", "jobs": self.snapshot(), "paused": PAUSED.is_set()})

    def set_paused(self, on):
        """Pause / resume. Paused: the running image waits at its next step and no waiting job starts."""
        with self._cv:
            if on:
                PAUSED.set()
            else:
                PAUSED.clear()
            self._broadcast()
            self._cv.notify_all()

    def put(self, job, front=False):
        with self._cv:
            if front:
                self._waiting.insert(0, job)
            else:
                self._waiting.append(job)
            self._broadcast()
            self._cv.notify()

    def get(self):
        with self._cv:
            while (not self._waiting or PAUSED.is_set()) and not self.closed:
                self._cv.wait(1.0 if PAUSED.is_set() else None)
            if not self._waiting:
                return None
            self.running = self._waiting.pop(0)
            if _engine is not None:
                _engine.cancel_event.clear()  # under the lock
                _engine.skip_event.clear()  # (same: a skip meant for an earlier job must not leak), so a cancel for this job can't be lost
            self._broadcast()
            return self.running

    def done(self):
        with self._cv:
            self.running = None
            self._broadcast()

    def cancel(self, job_id):
        """Returns 'removed' (was waiting), 'stopping' (was running) or None."""
        with self._cv:
            for j in self._waiting:
                if j["id"] == job_id:
                    self._waiting.remove(j)
                    emit({"type": "error", "id": job_id, "msg": "Removed from queue", "cancelled": True})
                    self._broadcast()
                    return "removed"
            if self.running and self.running["id"] == job_id:
                if _engine is not None:
                    _engine.cancel_event.set()
                return "stopping"
        return None

    def cancel_all(self):
        with self._cv:
            for j in self._waiting:
                emit({"type": "error", "id": j["id"], "msg": "Removed from queue", "cancelled": True})
            self._waiting.clear()
            if self.running and _engine is not None:
                _engine.cancel_event.set()
            PAUSED.clear()  # nothing left to pause
            self._broadcast()

    def busy(self):
        with self._cv:
            return bool(self.running or self._waiting)

    def busy_with_work(self):
        """True if anything other than a preload is running or waiting."""
        with self._cv:
            return any(j["cmd"] != "preload" for j in ([self.running] if self.running else []) + self._waiting)

    def make_way(self, model=None):
        """A real job arrived: drop waiting preloads, and stop a running one that loads a different model
        (one loading the same model, or a job that uses no model, lets it finish)."""
        with self._cv:
            for j in [j for j in self._waiting if j["cmd"] == "preload"]:
                self._waiting.remove(j)
                emit({"type": "result", "id": j["id"], "data": {"skipped": "replaced"}})
            r = self.running
            if r and r["cmd"] == "preload" and model is not None and r.get("model") != model and _engine is not None:
                _engine.cancel_event.set()
            self._broadcast()

    def close(self):
        with self._cv:
            self.closed = True
            self._cv.notify_all()


jobs = JobQueue()


_train_folders = set()  # training folders opened in this session (captions may only be written there)


def handle_quick(cmd, a):
    """Commands answered immediately on the reader thread."""
    if cmd == "ping":
        return "pong"
    if cmd == "system_info":
        return engine().system_info()
    if cmd == "list_models":
        all_marks = content.load_marks()
        marks, people = all_marks["models"], all_marks["people_models"]
        rows = engine().list_models()
        for r in rows:
            r["people"] = bool(people.get(r["id"]))
            r["nsfw_mark"] = marks.get(r["id"])
            r["nsfw"] = content.effective(r.get("nsfw_auto"), r["nsfw_mark"])
        return rows
    if cmd == "list_samplers":
        return samplers.list_for(a.get("family", "sd15"))
    if cmd == "list_loras":
        all_marks = content.load_marks()
        marks, people = all_marks["loras"], all_marks["people_loras"]
        rows = loras.all_loras(paths.LORAS_DIR)
        for r in rows:
            r["people"] = bool(people.get(r["name"]))
            r["nsfw_mark"] = marks.get(r["name"])
            r["nsfw"] = content.effective(r.get("nsfw_auto"), r["nsfw_mark"])
        return rows
    if cmd == "content_mark":  # your own NSFW mark on a model or LoRA (value: true, false or null = automatic)
        v = a.get("value")
        kind = "models" if a.get("kind") == "model" else "loras"
        if a.get("tag") == "people":  # "made of real people" tag (Hide People); value true = tagged
            kind = "people_" + kind
            v = True if v else None
        content.set_mark(kind, str(a.get("name")), None if v is None else bool(v))
        return True
    if cmd == "content_status":
        return {"checker_installed": content.checker_installed(), "checker_mb": int(content.CHECKER_GB * 1000),
                **library.nsfw_counts()}
    if cmd == "library_nsfw_mark":
        v = a.get("mark")
        return library.set_nsfw_mark(a.get("ids") or [], None if v is None else bool(v))
    if cmd == "set_settings":
        settings.update({k: v for k, v in a.items() if k in settings})
        return settings
    if cmd == "lora_triggers":
        name = a.get("name")
        row = next((l for l in loras.all_loras(paths.LORAS_DIR) if l["name"] == name), None)
        saved = (library.get_setting("lora_triggers", {}) or {}).get(name)
        if row is None:
            return {"found": [], "suggested": [], "saved": saved}
        original = diffusionbee.originals().get(row["file"]) if row["file"].endswith(".tdict") else None
        res = loras.find_triggers(row["file"], original)
        res["saved"] = saved
        return res
    if cmd == "lora_triggers_save":
        allt = library.get_setting("lora_triggers", {}) or {}
        words = [w.strip() for w in (a.get("words") or []) if str(w).strip()]
        if words:
            allt[a["name"]] = words
        else:
            allt.pop(a["name"], None)  # empty = go back to the automatically found words
        library.set_setting("lora_triggers", allt)
        return allt.get(a["name"])
    if cmd == "wildcards_list":
        import wildcards
        wildcards.folder().mkdir(parents=True, exist_ok=True)
        return {"folder": str(wildcards.folder()), "files": wildcards.list_files()}
    if cmd == "wildcards_preview":
        import random as _r
        import wildcards
        text = a.get("text") or ""
        return [wildcards.resolve(text, _r.Random()) for _ in range(int(a.get("count") or 4))]
    if cmd == "settings_reset":
        # Back to defaults. Never deletes images, models, LoRAs, presets, saved trigger words or the library.
        if jobs.busy():
            raise ValueError("Wait for the queue to finish first.")
        done = []
        if a.get("folders"):
            paths.IMAGES_DIR = paths.HOME / "images"
            paths.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            library.set_setting("images_dir", None)
            loras.EXTRA_DIRS[:] = []
            library.set_setting("lora_dirs", [])
            loras.FOLDER_FAMILY.clear()
            library.set_setting("lora_dir_families", {})
            model_hub.MODEL_DIRS[:] = []
            model_hub.invalidate()
            library.set_setting("model_dirs", [])
            done.append("folders")
        if a.get("settings"):
            net.strict = False
            library.set_setting("strict_offline", False)
            diffusionbee.enabled = True
            library.set_setting("import_diffusionbee", True)
            settings.update({"force_load": False, "low_memory": False, "image_format": "png", "jpeg_quality": 92,
                             "previews": True, "embed_meta": "details"})
            done.append("settings")
        return {"done": done, "images_dir": str(paths.IMAGES_DIR), "default_loras": str(paths.LORAS_DIR)}
    if cmd == "edit_start":
        import edits
        src_id = a.get("image_id")
        src_path = a.get("path")
        settings_ = None
        if src_id is not None:
            rec = library.get_image(int(src_id))
            if rec is None:
                raise ValueError("That picture isn't in the library any more.")
            src_path, settings_ = rec["path"], rec.get("settings")
        return {"file": edits.start(src_path, src_id, settings_, a.get("editor") or "PhotoCraft"),
                "folder": str(edits.folder())}
    if cmd == "image_b64":  # Adjust, when the window's own file reader isn't there (app not fully restarted)
        import base64
        import pictures
        p = str(a.get("path") or "")
        if not pictures.is_supported(p) or not os.path.isfile(p):
            raise ValueError("That picture wasn't found.")
        p = pictures.prepare(p)  # HEIC, TIFF, AVIF, RAW…: a PNG made by macOS, which the window can show
        if os.path.getsize(p) > 200 * 1024 * 1024:
            raise ValueError("That picture is too big to adjust.")
        with open(p, "rb") as fh:
            return {"data": base64.b64encode(fh.read()).decode("ascii")}
    if cmd == "adjust_save":  # Adjust (sliders and filters): the page wrote the finished PNG; add it to the Library
        import edits
        tmp_dir = os.path.realpath(str(paths.HOME / "adjust"))
        if a.get("data_b64"):  # (sent inline when the window's own file writer isn't there)
            import base64
            import uuid
            raw = base64.b64decode(a["data_b64"])
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("The adjusted picture isn't a PNG.")
            os.makedirs(tmp_dir, exist_ok=True)
            a["file"] = os.path.join(tmp_dir, f"adjust_{uuid.uuid4().hex[:8]}.png")
            with open(a["file"], "wb") as fh:
                fh.write(raw)
        f = os.path.realpath(str(a.get("file") or ""))
        if not f.startswith(tmp_dir + os.sep) or not os.path.isfile(f):
            raise ValueError("The adjusted picture wasn't found.")
        src_id = a.get("image_id")
        src_path = a.get("path")
        settings_ = {}
        if src_id is not None:
            src = library.get_image(int(src_id))
            if src:
                src_path = src["path"]
        adjust = a.get("adjust") if isinstance(a.get("adjust"), dict) else {}
        info = {"source_id": src_id, "source_path": src_path, "editor": "Adjust", "settings": settings_}
        try:
            rec = edits.import_into_library(library, paths.IMAGES_DIR, thumbs, f, info, emit,
                                            extra={"adjust": adjust}, suffix="adjusted")
        finally:
            try:
                os.remove(f)
            except OSError:
                pass
        return rec
    if cmd == "edit_check":
        import edits
        recs = edits.check(lambda f, info: edits.import_into_library(library, paths.IMAGES_DIR, thumbs, f, info, emit))
        return {"imported": recs, "pending": edits.pending()}
    if cmd == "edit_pending":
        import edits
        return {"pending": edits.pending(), "folder": str(edits.folder())}
    if cmd == "edit_finish":
        import edits
        return {"pending": edits.finish(a.get("name"))}
    if cmd == "export_image":  # Library → Batch edit & export: one picture at a time (the page shows progress)
        import batch_export
        tmp_dir = os.path.realpath(str(paths.HOME / "adjust"))
        rec = None
        if a.get("image_id") is not None:
            rec = library.get_image(int(a["image_id"]))
            if rec is None:
                raise ValueError("That picture isn't in the Library any more.")
        src = rec["path"] if rec else None
        tmp = a.get("tmp_file")
        if a.get("data_b64") and not tmp:  # (sent inline when the window's own file writer isn't there)
            import base64
            import uuid
            raw = base64.b64decode(a["data_b64"])
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("The adjusted picture isn't a PNG.")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp = os.path.join(tmp_dir, f"export_{uuid.uuid4().hex[:8]}.png")
            with open(tmp, "wb") as fh:
                fh.write(raw)
        if tmp:  # the page already applied an Adjust preset and wrote the result here
            tmp = os.path.realpath(str(tmp))
            if not tmp.startswith(tmp_dir + os.sep) or not os.path.isfile(tmp):
                raise ValueError("The adjusted picture wasn't found.")
        if not (tmp or (src and os.path.isfile(src))):
            raise ValueError(f"The picture file is missing: {src}")
        dest = str(a.get("dest") or "")
        if not os.path.isabs(dest) or re.search(r"\.app(/|$)", dest):
            raise ValueError("Choose the folder to save to.")
        settings_ = None
        if rec:
            settings_ = dict(rec.get("settings") or {})
            settings_.update({k: rec[k] for k in ("prompt", "negative_prompt", "model", "sampler", "steps", "cfg", "seed")
                              if k in rec})
            settings_["loras"] = rec.get("loras") or []
        try:
            info = batch_export.name_info(rec, src or tmp, int(a.get("n") or 1), a.get("preset") or "")
            opts = dict(a.get("opts") or {})
            opts["_orig_ext"] = os.path.splitext(src or tmp)[1]  # "same format" means the original's
            out = batch_export.export_one(tmp or src, dest, opts, info, settings_)
            if src and os.path.isfile(src):
                out["src_bytes"] = os.path.getsize(src)
            return out
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    if cmd == "image_prepare":  # a picture you dropped or picked: HEIC, TIFF, RAW… become a PNG the app can show
        import pictures
        out = pictures.prepare(a.get("path"))
        pictures.clear_cache()
        return {"path": out, "converted": out != str(a.get("path"))}
    if cmd == "library_get":
        rec = library.get_image(int(a.get("id")))
        if rec is None:
            raise ValueError("That picture isn't in the Library any more.")
        return rec
    if cmd == "export_presets_get":
        return [x for x in (library.get_setting("export_presets", []) or []) if isinstance(x, dict)]
    if cmd == "export_presets_set":
        presets = [{"name": str(x["name"]).strip()[:60] or "Preset", "set": x["set"]}
                   for x in (a.get("presets") or [])[:100]
                   if isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("set"), dict)]
        library.set_setting("export_presets", presets)
        return presets
    if cmd == "default_export_dir":
        d = Path.home() / "Pictures" / "Protoart Exports"
        return {"path": str(d)}
    if cmd == "adjust_presets_get":  # Adjust: your saved combinations of sliders and filters
        return [x for x in (library.get_setting("adjust_presets", []) or []) if isinstance(x, dict)]
    if cmd == "adjust_presets_set":
        presets = []
        for x in (a.get("presets") or [])[:200]:
            if isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("set"), dict):
                presets.append({"name": x["name"].strip()[:60] or "Preset", "set": x["set"]})
        library.set_setting("adjust_presets", presets)
        return presets
    if cmd == "lora_triggers_all":
        return library.get_setting("lora_triggers", {}) or {}
    if cmd == "list_embeddings":  # Create (above the negative prompt) and Settings → Embeddings
        return [{k: e.get(k) for k in ("name", "family", "file", "source", "vectors", "size_kb", "nsfw_auto")}
                for e in embeddings.all_embeddings()]
    if cmd == "embedding_dirs_get":
        return {"dirs": embeddings.EXTRA_DIRS, "default": embeddings.own_dir()}
    if cmd == "embedding_dirs_set":
        dirs = []
        for d in a.get("dirs") or []:
            if isinstance(d, str) and os.path.isabs(d) and d not in dirs and d != embeddings.own_dir():
                dirs.append(d)
        embeddings.EXTRA_DIRS[:] = dirs
        library.set_setting("embedding_dirs", dirs)
        return {"dirs": dirs, "default": embeddings.own_dir()}
    if cmd == "embedding_add_files":  # Settings → Embeddings → Add embedding files… (copies them in)
        files = [f for f in (a.get("files") or []) if isinstance(f, str) and os.path.isabs(f)]
        return embeddings.add_files(files[:200])
    if cmd == "lora_dirs_get":
        return {"dirs": loras.EXTRA_DIRS, "default": str(paths.LORAS_DIR), "families": dict(loras.FOLDER_FAMILY)}
    if cmd == "lora_dirs_set":
        dirs = []
        for d in a.get("dirs") or []:
            if isinstance(d, str) and os.path.isabs(d) and d not in dirs:
                dirs.append(d)
        loras.EXTRA_DIRS[:] = dirs
        library.set_setting("lora_dirs", dirs)
        keep = set(dirs) | {str(paths.LORAS_DIR)}  # a removed folder (and its sub-folders) forgets its type
        for k in [k for k in loras.FOLDER_FAMILY
                  if not any(k == d or k.startswith(d.rstrip("/") + "/") for d in keep)]:
            del loras.FOLDER_FAMILY[k]
        library.set_setting("lora_dir_families", dict(loras.FOLDER_FAMILY))
        return {"dirs": dirs, "default": str(paths.LORAS_DIR), "families": dict(loras.FOLDER_FAMILY)}
    if cmd == "model_family":  # a base model file's type, picked by you: SD 1.5 or SDXL (null = automatic)
        mid, fam = str(a.get("id") or ""), a.get("family")
        if ":" not in mid:
            raise ValueError("Only your own model files can be re-typed.")
        if fam in ("sd15", "sdxl"):
            model_hub.MODEL_FAMILY[mid] = fam
        else:
            model_hub.MODEL_FAMILY.pop(mid, None)
        library.set_setting("model_families", dict(model_hub.MODEL_FAMILY))
        if _engine is not None and _engine.loaded_key and _engine.loaded_key[0] == mid and not jobs.busy():
            _engine.unload()  # it loads as the new type next time
        return dict(model_hub.MODEL_FAMILY)
    if cmd == "lora_family":  # one LoRA's type, picked by you (family null = back to automatic)
        name, fam = str(a.get("name") or ""), a.get("family")
        if not name:
            raise ValueError("Which LoRA?")
        moved = None
        if fam in loras.FAMILIES and a.get("move", True):
            moved = move_lora_to_type_folder(name, fam)  # into the LoRA folder set to that type, if there is one
        if fam in loras.FAMILIES:
            loras.NAME_FAMILY[name] = fam
        else:
            loras.NAME_FAMILY.pop(name, None)
        library.set_setting("lora_families", dict(loras.NAME_FAMILY))
        return {"families": dict(loras.NAME_FAMILY), "moved": moved}
    if cmd == "lora_sort":  # Settings → LoRA folders → Sort into type folders (dry_run: just the plan)
        if not a.get("dry_run") and jobs.busy():
            raise ValueError("Wait for the queue to finish first (a LoRA may be in use).")
        return sort_loras(a.get("dir"), dry_run=bool(a.get("dry_run")))
    if cmd == "lora_dir_family":  # Settings → LoRA folders → "these are …": every LoRA in the folder is that type
        d, fam = a.get("dir"), a.get("family")
        if not isinstance(d, str) or not os.path.isabs(d):
            raise ValueError("Pick a folder first.")
        if fam in loras.FAMILIES:
            loras.FOLDER_FAMILY[d] = fam
        else:
            loras.FOLDER_FAMILY.pop(d, None)  # "detect automatically"
        library.set_setting("lora_dir_families", dict(loras.FOLDER_FAMILY))
        return {"dirs": loras.EXTRA_DIRS, "default": str(paths.LORAS_DIR), "families": dict(loras.FOLDER_FAMILY)}
    if cmd == "log_tail":
        try:
            with open(LOG_PATH, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - int(a.get("bytes") or 20000)))
                text = f.read().decode("utf-8", "replace")
                # the home folder (your username) is shown as ~ so the log can be shared safely
                return {"text": text.replace(str(Path.home()), "~"), "path": str(LOG_PATH)}
        except OSError:
            return {"text": "", "path": str(LOG_PATH)}
    if cmd == "log_clear":
        try:
            with open(LOG_PATH, "w"):
                pass
            old_log = LOG_PATH.with_suffix(".old.log")
            if old_log.exists():
                old_log.unlink()
        except OSError as e:
            raise ValueError(f"Couldn't clear the log: {e}")
        return True
    if cmd == "convert_list":
        out = []
        for i in diffusionbee.items(force=True):
            if i["kind"] in ("lora", "checkpoint", "embedding", "controlnet") and not i.get("error"):
                out.append({"file": i["file"], "name": i["name"], "kind": i["kind"], "family": i.get("family"),
                            "size_mb": i.get("size_mb")})
        return {"items": out, "default_out": str(paths.HOME / "converted"), "free_gb": round(
            __import__("shutil").disk_usage(paths.HOME).free / 2**30, 1)}
    if cmd == "select_object":
        import select_object
        from PIL import Image, ImageOps
        import pictures
        src = a.get("path") or ""
        if not os.path.isfile(src):
            raise ValueError("Couldn't open the picture.")
        image = ImageOps.exif_transpose(Image.open(pictures.prepare(src)))
        r = a.get("rect") or [0, 0, 1, 1]
        url, found = select_object.select(image, [float(v) for v in r[:4]], int(a.get("width") or image.width),
                                          int(a.get("height") or image.height))
        return {"mask": url, "found": found}
    if cmd == "list_vaes":
        import vaes
        return {"vaes": vaes.list_vaes(), "folder": str(paths.VAE_DIR)}
    if cmd == "list_controlnets":
        import controlnet as cnet
        return {"controlnets": cnet.user_files(), "folder": str(paths.CONTROLNET_DIR)}
    if cmd == "control_types":
        import controlnet as cnet
        return cnet.list_types()
    if cmd == "train_scan":
        import trainer
        items = trainer.scan_dataset(a.get("folder"))
        _train_folders.add(os.path.realpath(str(a.get("folder"))))
        return {"items": items, "suggested_steps": trainer.suggest_steps(len(items))}
    if cmd == "train_captions_save":
        import trainer
        # only caption files next to images in a training folder that was opened in the Train tab
        items = [it for it in (a.get("items") or []) if isinstance(it, dict)
                 and os.path.realpath(os.path.dirname(str(it.get("path", "")))) in _train_folders]
        return trainer.save_captions(items)
    if cmd in ("train_framing_save", "train_square_copies"):  # Train tab: how each picture is cropped to a square
        import trainer
        folder = os.path.realpath(str(a.get("folder")))
        if folder not in _train_folders:
            raise ValueError("Open the training folder in the Train tab first.")
        framing = a.get("framing") if isinstance(a.get("framing"), dict) else {}
        if cmd == "train_framing_save":
            return {"saved": trainer.save_framing(folder, framing)}
        res = int(a.get("size") or 512)
        if res not in (512, 768, 1024):
            raise ValueError("Pick 512, 768 or 1024.")
        trainer.save_framing(folder, framing)
        r = trainer.save_square_copies(folder, res, framing)
        _train_folders.add(os.path.realpath(r["folder"]))
        return r
    if cmd == "civitai_inspect":  # Settings → Models → From Civitai → Check
        if net.strict:
            raise ValueError("Strict offline mode is on (Settings → Privacy), so Civitai can't be checked.")
        net.ensure_downloads_work()  # route through curl when a firewall blocks Python (no other change)
        info = civitai.inspect(a.get("url"), library.get_setting("civitai_token"))
        info["destinations"] = civitai_destinations(info)
        try:  # "you may already have this"
            look = [d["path"] for d in info["destinations"]]
            if info.get("kind") == "lora":
                look += list(loras.EXTRA_DIRS) + [str(paths.LORAS_DIR)]
            elif info.get("kind") == "checkpoint":
                look += [str(paths.MODELS_DIR)] + list(model_hub.MODEL_DIRS)
            info["duplicates"] = civitai.find_duplicates(info, list(dict.fromkeys(look)))
        except Exception as e:
            log("duplicate check failed:", e)
            info["duplicates"] = []
        return info
    if cmd == "civitai_key_get":
        return {"set": bool(library.get_setting("civitai_token"))}
    if cmd == "civitai_key_set":  # kept in this Mac's library database; only ever sent to civitai.com
        key = str(a.get("key") or "").strip()
        if key and not re.fullmatch(r"[A-Za-z0-9_\-]{16,128}", key):
            raise ValueError("That doesn't look like a Civitai API key (civitai.com → Account settings → API keys).")
        library.set_setting("civitai_token", key or None)
        if key and a.get("check"):
            # check it with Civitai straight away (a signed-in-only request), so a wrong key shows up here
            net.ensure_downloads_work()
            try:
                civitai._get_json(civitai.API + "/models?limit=1&favorites=true", key)
            except ValueError as e:
                msg = str(e)
                if "log in" in msg:
                    msg = "Saved, but Civitai didn't accept this key: it may be mistyped or deleted. Copy it again from civitai.com (Account settings → API Keys)."
                return {"set": True, "ok": False, "msg": msg}
            return {"set": True, "ok": True}
        return {"set": bool(key)}
    if cmd == "hf_token_get":
        return {"set": bool(library.get_setting("hf_token"))}
    if cmd == "hf_token_set":  # kept in this Mac's library database; only ever sent to huggingface.co
        tok = str(a.get("token") or "").strip()
        if tok and not re.fullmatch(net.HF_TOKEN_RE, tok):
            raise ValueError("That doesn't look like a Hugging Face access token (they start with hf_; "
                             "huggingface.co → Settings → Access Tokens).")
        library.set_setting("hf_token", tok or None)
        net.apply_hf_token(tok or None)
        if tok and a.get("check") and not net.strict:
            was = net.is_online()
            net.set_online(True)
            try:
                from huggingface_hub import HfApi
                HfApi().whoami(token=tok)
            except Exception as e:
                bad = "401" in str(e) or "Unauthorized" in str(e) or "Invalid" in str(e)
                return {"set": True, "ok": False, "msg": "Saved, but Hugging Face didn't accept this token: it may be "
                        "mistyped or deleted. Copy it again (huggingface.co → Settings → Access Tokens)." if bad
                        else f"Saved. (Couldn't check it with Hugging Face right now: {str(e)[:120]})"}
            finally:
                net.set_online(was)
            return {"set": True, "ok": True}
        return {"set": bool(tok)}
    if cmd == "model_inspect":
        if net.strict:
            raise ValueError("Strict offline mode is on (Settings → Privacy), so Hugging Face can't be checked.")
        was = net.is_online()
        net.set_online(True)
        try:
            return model_hub.inspect(a.get("source"))
        finally:
            net.set_online(was)
    if cmd == "privacy_get":
        return {"strict": net.strict}
    if cmd == "privacy_set":
        net.strict = bool(a.get("strict"))
        library.set_setting("strict_offline", net.strict)
        if net.strict:
            net.set_online(False)
        return {"strict": net.strict}
    if cmd == "model_add":
        m = a.get("model") or {}
        keep = {k: m.get(k) for k in ("id", "repo", "file", "family", "class_name", "variant", "download_gb", "label")}
        if not keep["id"] or not keep["repo"] or keep["family"] not in model_hub.DEFAULTS:
            raise ValueError("Check the model first.")
        model_hub.USER_MODELS[:] = [x for x in model_hub.USER_MODELS if x.get("id") != keep["id"]] + [keep]
        library.set_setting("user_models", model_hub.USER_MODELS)
        return model_hub.USER_MODELS
    if cmd == "model_remove":
        model_hub.USER_MODELS[:] = [x for x in model_hub.USER_MODELS if x.get("id") != a.get("id")]
        library.set_setting("user_models", model_hub.USER_MODELS)
        return model_hub.USER_MODELS
    if cmd == "model_local_inspect":
        out, errors, lora_folders = [], [], []
        known = {os.path.realpath(d) for d in [str(paths.LORAS_DIR)] + list(loras.EXTRA_DIRS)}
        for pth in a.get("paths") or []:
            # a folder that also holds LoRAs (.safetensors under 1.5 GB, or DiffusionBee .tdict files) is
            # suggested as a LoRA folder, so its LoRAs show up too. For a picked file: the folder it's in.
            folder = pth if isinstance(pth, str) and os.path.isdir(pth) else (
                os.path.dirname(pth) if isinstance(pth, str) and os.path.isfile(pth) else None)
            if folder and os.path.realpath(folder) not in known and \
                    all(f["path"] != folder for f in lora_folders):
                n_lora = len(loras.scan(folder, only_loras=True)) + sum(
                    1 for i in diffusionbee.tdict_reader.scan([folder], recursive=True) if i.get("kind") == "lora")
                if n_lora:
                    lora_folders.append({"path": folder, "loras": n_lora})
            try:
                found, notes = model_hub.inspect_any(pth)  # a plain folder is searched for models inside it
                out += found
                errors += notes
            except Exception as e:
                errors.append(f"{os.path.basename(str(pth))}: {e}")
        if lora_folders:  # their LoRAs are taken care of: drop the "add it under LoRA folders" hints
            errors = [e for e in errors if "LoRA" not in e and "DiffusionBee" not in e]
        return {"found": out, "errors": errors, "lora_folders": lora_folders}
    if cmd == "model_local_add":
        keep_keys = ("id", "path", "kind", "family", "class_name", "label", "download_gb")
        for m in a.get("models") or []:
            entry = {k: m.get(k) for k in keep_keys}
            if not entry["id"] or not entry["path"] or entry["family"] not in model_hub.DEFAULTS:
                continue
            model_hub.LOCAL_MODELS[:] = [x for x in model_hub.LOCAL_MODELS if x.get("id") != entry["id"]] + [entry]
        library.set_setting("local_models", model_hub.LOCAL_MODELS)
        return model_hub.LOCAL_MODELS
    if cmd == "model_local_remove":
        model_hub.LOCAL_MODELS[:] = [x for x in model_hub.LOCAL_MODELS if x.get("id") != a.get("id")]
        library.set_setting("local_models", model_hub.LOCAL_MODELS)
        return model_hub.LOCAL_MODELS
    if cmd == "folders_get":
        return folder_roots()
    if cmd == "folder_relink":
        if jobs.busy():
            raise ValueError("Wait for the queue to finish first.")
        res = relink_folder(a.get("old"), a.get("new"))
        res["folders"] = folder_roots()
        return res
    if cmd == "model_dirs_get":
        return {"dirs": model_hub.MODEL_DIRS, "local": model_hub.LOCAL_MODELS}
    if cmd == "model_dirs_set":
        dirs = []
        for d in a.get("dirs") or []:
            if isinstance(d, str) and os.path.isabs(d) and d not in dirs:
                dirs.append(d)
        model_hub.MODEL_DIRS[:] = dirs
        model_hub.invalidate()
        library.set_setting("model_dirs", dirs)
        return {"dirs": dirs, "local": model_hub.LOCAL_MODELS}
    if cmd == "model_list_added":
        return model_hub.USER_MODELS
    if cmd == "diffusionbee_info":
        return diffusionbee.summary()
    if cmd == "diffusionbee_set":
        diffusionbee.enabled = bool(a.get("enabled"))
        library.set_setting("import_diffusionbee", diffusionbee.enabled)
        return diffusionbee.summary()
    if cmd == "library_query":
        rows = library.query(text=a.get("text"), loras=a.get("loras"), starred_only=a.get("starred_only", False),
                             model=a.get("model"), kind=a.get("kind"), limit=a.get("limit", 200),
                             offset=a.get("offset", 0), seed=a.get("seed"), hide_nsfw=bool(a.get("hide_nsfw")))
        thumbs.ensure(rows)
        return rows
    if cmd == "library_star":
        return library.set_starred(int(a["id"]), bool(a["starred"]))
    if cmd == "library_loras":
        return library.used_loras()
    if cmd == "library_delete":
        res = library.delete_images(a.get("ids") or [], include_versions=bool(a.get("include_versions")),
                                    delete_files=a.get("delete_files", True), dry_run=bool(a.get("dry_run")))
        if not a.get("dry_run"):
            thumbs.remove(res["ids"])
        return res
    if cmd == "free_original":
        if jobs.busy():
            raise ValueError("Wait for the queue to finish first.")
        return engine().free_original(a.get("model"))
    if cmd == "images_dir_get":
        return {"path": str(paths.IMAGES_DIR), "default": str(paths.HOME / "images")}
    if cmd == "downloads_dir_get":
        import shutil as _sh
        cur = paths.hub_cache()
        free = None
        try:
            free = round(_sh.disk_usage(cur if cur.exists() else cur.parent).free / 1e9, 1)
        except OSError:
            pass
        return {"path": str(cur), "default": str(paths.DEFAULT_HUB_CACHE), "custom": paths.configured_hub_cache() is not None,
                "missing": paths.HUB_CACHE_MISSING, "free_gb": free}
    if cmd == "downloads_dir_size":
        return {"gb": round(_folder_bytes(Path(a.get("path") or paths.hub_cache())) / 1e9, 1)}
    if cmd == "downloads_dir_set":
        new = a.get("path")
        if new:
            p = Path(new).expanduser()
            if not p.is_absolute():
                raise ValueError("Pick a folder.")
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".protoart_write_test"
            try:
                probe.write_text("ok")
                probe.unlink()
            except OSError:
                raise ValueError(f"Can't write to {p}: pick a folder you can save files in.")
            paths.save_hub_cache(p)
        else:
            paths.save_hub_cache(None)
        return {"path": str(p) if new else str(paths.DEFAULT_HUB_CACHE), "restart": True}

        return set_images_dir(a.get("path"), bool(a.get("move_existing")))
    if cmd == "presets_list":
        return library.list_presets()
    if cmd == "presets_save":
        return library.save_preset(a.get("name"), a.get("settings") or {})
    if cmd == "presets_delete":
        return library.delete_preset(a.get("name"))
    raise KeyError(cmd)


def relink_folder(old, new):
    """You moved a folder (e.g. to another drive): point everything that used it at the new place.
    LoRA folders and their assigned types, watched model folders, models you added, your NSFW / People marks,
    the Library's record of which model made each picture, and the image folder are all updated."""
    old = os.path.normpath(str(old or "")).rstrip("/")
    new = os.path.normpath(str(new or "")).rstrip("/")
    if not old or not os.path.isabs(old) or not os.path.isabs(new):
        raise ValueError("Pick the folder's new location.")
    if not os.path.isdir(new):
        raise ValueError(f"{new} isn't a folder (is the drive connected?).")
    if old == new:
        return {"changed": 0, "pictures": 0, "images": 0, "old": old, "new": new}

    def move(p):
        """old/… -> new/…; anything else unchanged."""
        if not isinstance(p, str):
            return p
        for pre in ("local:", ""):
            if p.startswith(pre):
                body = p[len(pre):]
                if body == old or body.startswith(old + "/"):
                    return pre + new + body[len(old):]
        return p

    # Guard against picking the wrong folder: something you set up must actually be there.
    known = [m.get("path") for m in model_hub.LOCAL_MODELS] + list(loras.EXTRA_DIRS) + list(model_hub.MODEL_DIRS) + \
        list(embeddings.EXTRA_DIRS)
    inside = [p for p in known if isinstance(p, str) and p.startswith(old + "/")]  # (the folder itself proves nothing)
    if inside and not any(os.path.exists(move(p)) for p in inside):
        raise ValueError(f"None of your models or folders from {os.path.basename(old)} are in {new}. "
                         f"Pick the folder that is now “{os.path.basename(old)}” (the same folder, in its new place).")

    changed = 0

    def fix_list(lst):
        nonlocal changed
        out = []
        for x in lst:
            y = move(x)
            changed += y != x
            if y not in out:
                out.append(y)
        return out

    loras.EXTRA_DIRS[:] = fix_list(list(loras.EXTRA_DIRS))
    library.set_setting("lora_dirs", list(loras.EXTRA_DIRS))
    embeddings.EXTRA_DIRS[:] = fix_list(list(embeddings.EXTRA_DIRS))
    library.set_setting("embedding_dirs", list(embeddings.EXTRA_DIRS))
    fam = {move(k): v for k, v in loras.FOLDER_FAMILY.items()}
    changed += sum(1 for k in loras.FOLDER_FAMILY if move(k) != k)
    loras.FOLDER_FAMILY.clear()
    loras.FOLDER_FAMILY.update(fam)
    library.set_setting("lora_dir_families", dict(fam))
    model_hub.MODEL_DIRS[:] = fix_list(list(model_hub.MODEL_DIRS))
    library.set_setting("model_dirs", list(model_hub.MODEL_DIRS))
    for m in model_hub.LOCAL_MODELS:
        for k in ("id", "path"):
            if move(m.get(k)) != m.get(k):
                m[k] = move(m[k])
                changed += 1
    library.set_setting("local_models", model_hub.LOCAL_MODELS)
    model_hub.invalidate()
    fams = {move(k): v for k, v in model_hub.MODEL_FAMILY.items()}
    changed += sum(1 for k in model_hub.MODEL_FAMILY if move(k) != k)
    model_hub.MODEL_FAMILY.clear()
    model_hub.MODEL_FAMILY.update(fams)
    library.set_setting("model_families", dict(fams))
    marks = content.load_marks()
    for kind, d in marks.items():
        for k in list(d):
            if move(k) != k:
                content.set_mark(kind, move(k), d[k])
                content.set_mark(kind, k, None)
                changed += 1
    pictures = library.relink_models(lambda m: move(m))
    changed += library.relink_preset_models(lambda m: move(m))
    images = library.rewrite_path_prefix(old, new)  # pictures saved inside the moved folder
    if isinstance(paths.IMAGES_DIR, Path) and move(str(paths.IMAGES_DIR)) != str(paths.IMAGES_DIR):
        paths.IMAGES_DIR = Path(move(str(paths.IMAGES_DIR)))
    return {"changed": changed, "pictures": pictures, "images": images, "old": old, "new": new}


# Sub-folder names recognised as "this type" when sorting (case-insensitive); new ones get the first-column name.
TYPE_FOLDER_NAMES = {"sd15": ("sd 1.5", "sd1.5", "sd15", "sd1", "sd-1.5", "sd_1.5", "1.5", "sd 1"),
                     "sdxl": ("sdxl", "xl", "sd xl"), "flux": ("flux", "flux.1", "flux1"),
                     "flux2": ("flux.2", "flux2", "flux 2", "flux2 klein", "klein"), "ernie": ("ernie",)}
TYPE_FOLDER_NEW = {"sd15": "SD 1.5", "sdxl": "SDXL", "flux": "FLUX", "flux2": "FLUX.2", "ernie": "ERNIE"}
SIDECAR_SKIP = (".safetensors", ".tdict", ".ckpt")


def _sidecars(src):
    folder, stem = os.path.dirname(src), os.path.splitext(os.path.basename(src))[0]
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return [os.path.join(folder, f) for f in names
            if f != os.path.basename(src) and f.startswith(stem + ".") and not f.endswith(SIDECAR_SKIP)]


def sort_loras(root, dry_run=True):
    """Sort the LoRAs in one LoRA folder into a sub-folder per model type (SD 1.5 / SDXL / FLUX / ERNIE).
    Uses sub-folders you already have (e.g. 'Flux', 'sdxl', 'SD1'), creates the missing ones, sets each one's
    type, and moves every LoRA whose type is known (with its trigger-word / preview files). LoRAs of unknown
    type and base models stay where they are; nothing is ever overwritten."""
    import shutil
    if not isinstance(root, str) or not os.path.isdir(root):
        raise ValueError("Pick one of your LoRA folders.")
    root = root.rstrip("/")
    real_root = os.path.realpath(root)
    # existing sub-folders by type: one you already labelled wins, then a recognised name
    folders = {}
    for d, f in loras.FOLDER_FAMILY.items():
        if f in TYPE_FOLDER_NEW and os.path.dirname(d.rstrip("/")) == root and os.path.isdir(d):
            folders.setdefault(f, d)
    try:
        subdirs = sorted(e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e)))
    except OSError as e:
        raise ValueError(f"Can't read {root}: {e.strerror}")
    for fam, names in TYPE_FOLDER_NAMES.items():
        if fam not in folders:
            hit = next((e for e in subdirs if e.strip().lower() in names), None)
            if hit:
                folders[fam] = os.path.join(root, hit)
    plan_folders = {fam: {"path": folders.get(fam) or os.path.join(root, TYPE_FOLDER_NEW[fam]),
                          "exists": fam in folders} for fam in TYPE_FOLDER_NEW}
    moves, unknown, skipped = [], [], []
    db_root = os.path.realpath(str(diffusionbee.ROOT))
    for row in loras.all_loras(paths.LORAS_DIR):
        src = os.path.realpath(row.get("file") or "")
        if not src.startswith(real_root + os.sep) or src.startswith(db_root + os.sep):
            continue
        fam = row["family"] if row.get("family_set") else row.get("detected_family", row.get("family"))
        if fam not in TYPE_FOLDER_NEW:
            unknown.append(row["name"])
            continue
        dest_dir = plan_folders[fam]["path"]
        if os.path.dirname(src) == os.path.realpath(dest_dir):
            continue  # already sorted
        clash = next((f for f in [src] + _sidecars(src)
                      if os.path.exists(os.path.join(dest_dir, os.path.basename(f)))), None)
        if clash:
            skipped.append({"name": row["name"], "why": f"{os.path.basename(clash)} is already in {os.path.basename(dest_dir)}"})
            continue
        moves.append({"name": row["name"], "family": fam, "from": src, "to": dest_dir})
    counts = {}
    for m in moves:
        counts[m["family"]] = counts.get(m["family"], 0) + 1
    out = {"dir": root, "folders": plan_folders, "moves": len(moves), "by_type": counts,
           "unknown": unknown, "skipped": skipped, "done": False}
    if dry_run:
        return out
    used = {m["family"] for m in moves} | {f for f, v in plan_folders.items() if v["exists"]}
    for fam in used:  # make the folders that are needed and label them with their type
        d = plan_folders[fam]["path"]
        os.makedirs(d, exist_ok=True)
        loras.FOLDER_FAMILY[d] = fam
    library.set_setting("lora_dir_families", dict(loras.FOLDER_FAMILY))
    moved, errors = 0, []
    for m in moves:
        try:
            for f in [m["from"]] + _sidecars(m["from"]):
                shutil.move(f, os.path.join(m["to"], os.path.basename(f)))
            moved += 1
        except OSError as e:
            errors.append(f"{m['name']}: {e.strerror}")
    out.update(done=True, moved=moved, errors=errors)
    return out


def merge_loras(a, emit_progress=None):
    """Train tab → Merge LoRAs. a: {"items": [{"name", "weight"}], "name", "mode": "exact"|"smaller", "rank"}.
    Only LoRAs of one model type, as .safetensors. The result goes into Protoart's LoRA folder for that type."""
    import lora_merge
    rows = {l["name"]: l for l in loras.all_loras(paths.LORAS_DIR)}
    picked = []
    for it in (a.get("items") or []):
        row = rows.get(it.get("name"))
        if not row:
            raise ValueError(f"LoRA not found: {it.get('name')}")
        picked.append((row, it))
    fams = {r.get("family") for r, _ in picked}
    if len(fams) != 1 or None in fams or "unknown" in fams:
        raise ValueError("Pick LoRAs made for the same model type (all SD 1.5, all SDXL…); LoRAs of unknown type "
                         "need a type first (Models tab → Fix a type).")
    fam = fams.pop()
    name = lora_merge.safe_name(a.get("name"))
    dest_dir = next((d for d, f in loras.FOLDER_FAMILY.items() if f == fam
                     and os.path.dirname(d.rstrip("/")) == str(paths.LORAS_DIR) and os.path.isdir(d)), str(paths.LORAS_DIR))
    out = Path(dest_dir) / f"{name}.safetensors"
    if out.exists() or name in rows:
        raise ValueError(f"A LoRA called '{name}' already exists. Pick another name.")
    items = [{"path": r["file"], "weight": float(it.get("weight", 1.0)), "name": r["name"]} for r, it in picked]
    try:
        res = lora_merge.merge(items, out, mode="smaller" if a.get("mode") == "smaller" else "exact",
                               rank=int(a.get("rank") or 32), emit=emit_progress, cancelled=engine_cancelled)
    except lora_merge.MergeError as e:
        raise ValueError(str(e))
    res["family"] = fam
    return res


def make_default_type_folders():
    """First run: Protoart's own LoRA folder starts with one sub-folder per model type, already labelled,
    so new LoRAs can be dropped straight into the right one."""
    if library.get_setting("default_type_folders_made"):
        return
    try:
        for fam, name in TYPE_FOLDER_NEW.items():
            d = os.path.join(str(paths.LORAS_DIR), name)
            if not any(v == fam and os.path.dirname(k.rstrip("/")) == str(paths.LORAS_DIR) for k, v in loras.FOLDER_FAMILY.items()):
                os.makedirs(d, exist_ok=True)
                loras.FOLDER_FAMILY.setdefault(d, fam)
        library.set_setting("lora_dir_families", dict(loras.FOLDER_FAMILY))
        library.set_setting("default_type_folders_made", True)
    except OSError as e:
        log("couldn't make the LoRA type folders:", e)


def civitai_destinations(info):
    """Where a Civitai download can go, best first: LoRAs into a folder set to their type (yours before
    Protoart's own), checkpoints into your checkpoints folder or a watched models folder."""
    out = []
    if info.get("kind") == "lora":
        fam = info.get("family")
        app_dir = str(paths.LORAS_DIR)
        typed = [d for d, f in loras.FOLDER_FAMILY.items() if f == fam and os.path.isdir(d)]
        typed.sort(key=lambda d: d.startswith(app_dir))  # your own folders first
        out += [{"path": d, "label": f"{os.path.basename(d)} ({civitai_family_label(fam)} LoRAs)"} for d in typed]
        for d in list(loras.EXTRA_DIRS) + [app_dir]:
            if os.path.isdir(d) and all(o["path"] != d for o in out):
                out.append({"path": d, "label": os.path.basename(d.rstrip("/")) + (" (Protoart's LoRA folder)" if d == app_dir else "")})
    elif info.get("kind") == "embedding":
        out.append({"path": embeddings.own_dir(), "label": "Protoart's embeddings folder"})
        out += [{"path": d, "label": os.path.basename(d.rstrip("/")) + " (your embeddings folder)"}
                for d in embeddings.EXTRA_DIRS if os.path.isdir(d)]
    elif info.get("kind") == "vae":
        out.append({"path": str(paths.VAE_DIR), "label": "Protoart's VAE folder"})
    elif info.get("kind") == "controlnet":
        out.append({"path": str(paths.CONTROLNET_DIR), "label": "Protoart's ControlNet folder"})
    elif info.get("kind") == "checkpoint":
        out.append({"path": str(paths.MODELS_DIR), "label": "Your checkpoints (Protoart)"})
        out += [{"path": d, "label": os.path.basename(d.rstrip("/")) + " (watched models folder)"}
                for d in model_hub.MODEL_DIRS if os.path.isdir(d)]
    return out


def civitai_family_label(fam):
    return {"sd15": "SD 1.5", "sdxl": "SDXL", "flux": "FLUX", "flux2": "FLUX.2", "ernie": "ERNIE"}.get(fam, "other")


def civitai_job(a, eng):
    """Download a model/LoRA from Civitai into the chosen folder (Settings → Models → From Civitai)."""
    if net.strict:
        raise ValueError("Strict offline mode is on (Settings → Privacy): turn it off to download from Civitai.")
    token = library.get_setting("civitai_token")
    net.set_online(True)
    try:
        info = civitai.inspect(a.get("url"), token)
        dests = [d["path"] for d in civitai_destinations(info)]
        dest = a.get("dest") if a.get("dest") in dests else (dests[0] if dests else None)
        if not dest:
            raise ValueError("No folder to save it in: add a LoRA folder or a models folder first.")
        eng.emit({"type": "status", "msg": f"Downloading {info['name']} from Civitai…"})

        def progress(got, total):
            if total:
                eng.emit({"type": "progress", "image": 1, "images": 1, "step": int(got / 1e6), "steps": max(1, int(total / 1e6))})
            eng.emit({"type": "status", "msg": f"Downloading {info['name']}: {got / 1e6:.0f}"
                                               + (f" of {total / 1e6:.0f} MB" if total else " MB")})
        try:
            path = civitai.download(info, dest, token, progress, cancelled=eng.cancel_event.is_set)
        except InterruptedError:
            from engine import Cancelled
            raise Cancelled()
    finally:
        net.set_online(False)
    name = os.path.splitext(os.path.basename(path))[0]
    if info["kind"] == "embedding" and not embeddings.is_embedding_file(path):
        try:
            os.remove(path)  # our own download, not an embedding after all
        except OSError:
            pass
        raise ValueError(f"{os.path.basename(path)} from Civitai isn't an embedding the app can read, so it wasn't kept.")
    if info.get("nsfw") and info["kind"] in ("lora", "checkpoint"):  # Civitai marks it NSFW: so does the content filter (your own mark still wins)
        kind, key = ("loras", name) if info["kind"] == "lora" else ("models", None)
        if info["kind"] == "checkpoint":
            key = ("file:" + os.path.basename(path)) if dest == str(paths.MODELS_DIR) else ("local:" + path)
        if key and key not in content.load_marks()[kind]:
            content.set_mark(kind, key, True)
    model_hub.invalidate()
    eng.emit({"type": "status", "msg": f"{info['name']} downloaded to {dest}"})
    return {"path": path, "name": name, "kind": info["kind"], "family": info.get("family"), "dest": dest}


def move_lora_to_type_folder(name, fam):
    """Move a LoRA file (and its sidecar files: trigger words, preview, info) into the LoRA folder you set to
    type `fam`. Only from your own LoRA folders; never overwrites. Returns {"to": folder} / {"skipped": why}."""
    import shutil
    row = next((l for l in loras.all_loras(paths.LORAS_DIR) if l["name"] == name), None)
    if row is None or not row.get("file"):
        return {"skipped": "LoRA not found"}
    src = os.path.realpath(row["file"])
    own = [os.path.realpath(d) for d in [str(paths.LORAS_DIR)] + list(loras.EXTRA_DIRS)]
    if not any(src.startswith(d + os.sep) for d in own):
        return {"skipped": "it isn't in one of your LoRA folders (DiffusionBee's own files are left where they are)"}
    # your folders stay yours: a LoRA in your own folders never moves into the app's built-in LoRA folder
    # (e.g. from your external drive onto the Mac), and one in the built-in folder stays in it
    app_dir = os.path.realpath(str(paths.LORAS_DIR))
    in_app = src.startswith(app_dir + os.sep)

    def same_side(d):
        r = os.path.realpath(d)
        return (r == app_dir or r.startswith(app_dir + os.sep)) == in_app
    targets = [d for d, f in loras.FOLDER_FAMILY.items() if f == fam and os.path.isdir(d) and same_side(d)]
    if not targets:
        return {"skipped": "none of your LoRA folders is set to that type yet: use “Sort into type folders” "
                           "(Settings → LoRA folders) to make them"}
    if any(src.startswith(os.path.realpath(t) + os.sep) for t in targets):
        return {"skipped": "already in a folder of that type"}
    # several folders of that type (e.g. yours and Protoart's own): the one closest to the LoRA's current place
    dest_dir = max(targets, key=lambda t: len(os.path.commonpath([src, os.path.realpath(t)])))
    stem, folder = os.path.splitext(os.path.basename(src))[0], os.path.dirname(src)
    files = [src] + [os.path.join(folder, f) for f in os.listdir(folder)
                     if f != os.path.basename(src) and (f.startswith(stem + ".") and not f.endswith((".safetensors", ".tdict")))]
    clash = [f for f in files if os.path.exists(os.path.join(dest_dir, os.path.basename(f)))]
    if clash:
        return {"skipped": f"{os.path.basename(clash[0])} already exists in {dest_dir}"}
    for f in files:
        shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))
    return {"to": dest_dir, "files": len(files)}


def folder_roots():
    """The folders you've told the app about, and whether each one is there right now (Settings → Moved a folder?)."""
    roots = []
    for d in list(loras.EXTRA_DIRS) + list(model_hub.MODEL_DIRS) + list(embeddings.EXTRA_DIRS) + \
            [os.path.dirname(m.get("path") or "") for m in model_hub.LOCAL_MODELS]:
        d = (d or "").rstrip("/")
        if d and d not in roots and not any(d.startswith(r + "/") for r in roots):
            roots = [r for r in roots if not r.startswith(d + "/")] + [d]
    return [{"path": r, "found": os.path.isdir(r)} for r in roots]


def set_images_dir(new, move_existing):
    if jobs.busy():
        raise ValueError("Wait for the queue to finish before changing the image folder.")
    new = Path(str(new or "")).expanduser()
    if not new.is_absolute():
        raise ValueError("Choose a folder")
    new.mkdir(parents=True, exist_ok=True)
    probe = new / ".beestudio_write_test"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise ValueError(f"Can't save images in {new}: {e.strerror}")
    old = paths.IMAGES_DIR
    result = {"path": str(new), "moved": 0, "failed": []}
    if move_existing and Path(old).resolve() != new.resolve():
        result.update(library.relocate(old, new))
    paths.IMAGES_DIR = new
    library.set_setting("images_dir", str(new))
    return result


def reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log("bad message", line[:200])
            continue
        mid, cmd, a = msg.get("id"), msg.get("cmd"), msg.get("args") or {}
        if cmd == "cancel_job":
            emit({"type": "result", "id": mid, "data": jobs.cancel(a.get("job_id"))})
            continue
        if cmd == "pause":  # Pause / Resume generating (args: {"on": true|false})
            jobs.set_paused(bool(a.get("on", True)))
            emit({"type": "result", "id": mid, "data": {"paused": PAUSED.is_set()}})
            continue
        if cmd == "skip_image":  # stop just the image being made now; the job goes on to its next image
            if _engine is not None and jobs.running and jobs.running.get("cmd") == "generate":
                _engine.skip_event.set()
            emit({"type": "result", "id": mid, "data": True})
            continue
        if cmd in ("cancel", "cancel_all"):  # "cancel" kept for compatibility: stops everything
            jobs.cancel_all()
            emit({"type": "result", "id": mid, "data": True})
            continue
        if cmd == "queue_get":
            emit({"type": "result", "id": mid, "data": jobs.snapshot()})
            continue
        if cmd == "preload" and jobs.busy_with_work():
            emit({"type": "result", "id": mid, "data": {"skipped": "busy"}})  # only uses idle time
            continue
        if cmd in SIDE_LANE and jobs.running:  # quick Tools jobs don't wait behind a long generation
            threading.Thread(target=run_side, args=(mid, cmd, a), daemon=True).start()
            continue
        if cmd in HEAVY:
            job = {"id": mid, "cmd": cmd, "args": a, "label": job_label(cmd, a), "model": a.get("model")}
            jobs.make_way(None if cmd == "preload" else (a.get("model") or (a.get("base") or {}).get("model")))
            # Image -> prompt is quick and the user is waiting on it, so it jumps the queue.
            jobs.put(job, front=cmd in ("interrogate", "enhance_prompt", "control_preview"))
            continue
        if cmd in NET_QUICK:  # these wait on the internet: never hold up the other buttons while they do
            threading.Thread(target=run_quick, args=(mid, cmd, a), daemon=True).start()
            continue
        run_quick(mid, cmd, a)
    jobs.close()  # stdin closed: the app quit


# Tools jobs that run on the processor (not the graphics chip) and take seconds: while a picture is being made they
# run beside it instead of waiting in the queue (Remove with smart fill uses LaMa on the CPU; quick fill is OpenCV).
SIDE_LANE = ("heal",)


def run_side(mid, cmd, a):
    from engine import UserError
    def side_emit(m):
        emit(dict(m, job=mid))
    try:
        eng = engine()
        if cmd == "heal":
            data = eng.heal(a, emit=side_emit)
        else:
            raise KeyError(cmd)
        emit({"type": "result", "id": mid, "data": data})
    except UserError as e:
        emit({"type": "error", "id": mid, "msg": str(e)})
    except Exception as e:
        log(traceback.format_exc())
        emit({"type": "error", "id": mid, "msg": f"{type(e).__name__}: {e}"})


# quick commands that talk to a website (up to ~25 s if it's slow or blocked): run beside the reader, so the
# app stays responsive (an unanswered Check used to leave every other button waiting)
NET_QUICK = ("select_object", "civitai_inspect", "civitai_key_set", "hf_token_set", "model_inspect",
             "train_square_copies")  # (that one isn't online, but can take a while for a big folder)


def run_quick(mid, cmd, a):
    try:
        emit({"type": "result", "id": mid, "data": handle_quick(cmd, a)})
    except KeyError:
        emit({"type": "error", "id": mid, "msg": f"Unknown command {cmd}"})
    except Exception as e:
        log(traceback.format_exc())
        emit({"type": "error", "id": mid, "msg": str(e)})


def convert_tdict_job(a, job_id):
    """Convert DiffusionBee .tdict files to .safetensors (numpy only; the originals are kept)."""
    import tdict_reader
    roots = diffusionbee.allowed_roots()
    files = [f for f in (a.get("files") or []) if isinstance(f, str) and f.endswith(".tdict")
             and any(os.path.realpath(f).startswith(r + os.sep) for r in roots)]  # DiffusionBee's or your folders
    out_dir = a.get("out_dir") or str(paths.HOME / "converted")
    lora_dir = str(paths.LORAS_DIR) if a.get("loras_to_lora_folder", True) else out_dir
    done, errors = [], []
    for i, f in enumerate(files):
        if _engine is not None and _engine.cancel_event.is_set():
            break
        name = os.path.basename(f)
        emit({"type": "status", "msg": f"Converting {i + 1}/{len(files)}: {name}…", "job": job_id})
        emit({"type": "progress", "image": 1, "images": 1, "step": i, "steps": len(files), "job": job_id})
        try:
            kind = tdict_reader.describe(f)["kind"]
            dest = lora_dir if kind == "lora" else out_dir
            done.append({"from": f, "to": tdict_reader.convert_to_safetensors(f, dest), "kind": kind})
        except Exception as e:
            errors.append(f"{name}: {e}")
    return {"converted": done, "errors": errors, "out_dir": out_dir}


def sfw_check(kind, a):
    """Safe-for-work mode: refuse prompts that ask for adult content, and LoRAs or models marked (or named) NSFW."""
    if not settings.get("sfw_mode") or kind not in ("generate", "edit_words", "enhance_prompt"):
        return
    from engine import UserError
    off = " (Safe-for-work mode is on; switch it off with 🛡 SFW at the top of Settings.)"
    texts = a.get("prompt")
    for t in (texts if isinstance(texts, list) else [texts]):
        if content.prompt_flag(t):
            raise UserError("This prompt asks for adult content, so it wasn't made." + off)
    marks = content.load_marks()
    names = [l.get("name") for l in (a.get("loras") or []) if isinstance(l, dict)]
    if names:
        rows = {r["name"]: r for r in loras.all_loras(paths.LORAS_DIR)}
        for n in names:
            mark = (marks.get("loras") or {}).get(n)
            auto = bool(rows.get(n, {}).get("nsfw_auto")) or content.name_flag(n)
            if mark is True or (mark is None and auto):
                raise UserError(f"The LoRA “{n}” is marked NSFW, so it can't be used." + off)
    mid = a.get("model")
    if kind == "generate" and mid:
        mark = (marks.get("models") or {}).get(mid)
        if mark is True or (mark is None and content.name_flag(mid)):
            raise UserError("This model is marked NSFW, so it can't be used." + off)


def run_job(kind, a):
    sfw_check(kind, a)
    if kind == "convert_tdict":
        return convert_tdict_job(a, jobs.running["id"] if jobs.running else None)
    eng = engine()
    if kind == "downloads_move":
        def prog(done, total, msg):
            eng.emit({"type": "progress", "image": 1, "images": 1, "step": int(done / 1e6), "steps": int(total / 1e6)})
            eng.emit({"type": "status", "msg": msg})
        return move_downloads(a, prog)
    if kind == "model_download":
        return eng.download_model(a.get("model"))
    if kind == "preload":
        return eng.preload(a, force=settings["force_load"], low_memory=settings["low_memory"])
    if kind == "nsfw_scan":
        return eng.nsfw_scan(a)
    if kind == "model_delete":
        return eng.delete_model(a)
    if kind == "civitai_download":
        return civitai_job(a, eng)
    if kind == "postprocess":
        return eng.postprocess(a, force=settings["force_load"])
    if kind == "interrogate":
        return eng.interrogate(a)
    if kind == "enhance_prompt":
        return eng.enhance_prompt(a)
    if kind == "compare":
        return eng.compare(a, force=settings["force_load"], low_memory=settings["low_memory"])
    if kind == "train_lora":
        return eng.train_lora(a)
    if kind == "train_embedding":
        if os.path.realpath(str(a.get("folder"))) not in _train_folders:
            raise ValueError("Open the training folder in the Train tab first.")
        return eng.train_embedding(a)
    if kind == "lora_merge":
        return merge_loras(a, emit_progress=eng.emit)
    if kind == "train_autocaption":
        if os.path.realpath(str(a.get("folder"))) not in _train_folders:
            raise ValueError("Open the training folder in the Train tab first.")
        return eng.train_autocaption(a)
    if kind == "control_preview":
        return eng.control_preview(a)
    if kind == "heal":
        return eng.heal(a)
    if kind == "vectorize":
        return eng.vectorize(a)
    if kind == "edit_words":
        return eng.edit_words(a)
    if kind == "vae_get":
        import vaes
        from engine import UserError
        if net.strict:
            raise UserError("Strict offline mode is on (Settings → Privacy): turn it off to download the VAE.")
        return vaes.get_recommended(a.get("family"), emit=eng.emit)
    return eng.generate(a, force=settings["force_load"], low_memory=settings["low_memory"])


def main():
    make_default_type_folders()
    try:
        os.makedirs(embeddings.own_dir(), exist_ok=True)  # Settings → Embeddings: drop embedding files in here
    except OSError:
        pass
    threading.Thread(target=reader, daemon=True).start()
    emit({"type": "ready"})
    from engine import Cancelled, UserError
    while True:
        job = jobs.get()
        if job is None:
            break
        mid, kind, a = job["id"], job["cmd"], job["args"]
        eng = None
        try:
            eng = engine()
            eng.current_job = mid
            eng.output_format = ("jpeg" if settings["image_format"] == "jpeg" else "png",
                                 max(50, min(100, int(settings["jpeg_quality"] or 92))))
            eng.previews = bool(settings.get("previews", True))
            eng.embed_meta = "none" if settings.get("embed_meta") == "none" else "details"
            eng.check_pictures = bool(settings.get("check_pictures") or settings.get("sfw_mode"))
            try:
                data = run_job(kind, a)  # offline first: uses only what's already on this Mac
            except Cancelled:
                raise
            except Exception as first:
                if kind in ("preload", "model_delete", "civitai_download") or not net.looks_like_missing_download(first):
                    raise  # a preload never downloads
                if net.strict:
                    raise UserError("This needs files that aren't on your Mac yet, and Strict offline mode is on. "
                                    "Turn it off in Settings → Privacy to allow the one-time download.")
                log("needs a download; retrying online:", first)
                emit({"type": "status", "msg": "Downloading files this needs from Hugging Face (first time only)…", "job": mid})
                net.set_online(True)
                try:
                    data = run_job(kind, a)
                finally:
                    net.set_online(False)
            emit({"type": "result", "id": mid, "data": data})
        except Cancelled:
            emit({"type": "error", "id": mid, "msg": "Stopped", "cancelled": True})
        except UserError as e:
            emit({"type": "error", "id": mid, "msg": str(e)})
        except Exception as e:
            log(traceback.format_exc())
            problem = net.hf_auth_problem(e)  # a gated Hugging Face model: say how to sign in
            emit({"type": "error", "id": mid, "msg": problem or f"{type(e).__name__}: {e}"})
        finally:
            if eng is not None:
                eng.current_job = None
                try:
                    getattr(eng, "release_memory", lambda: None)()
                except Exception:
                    log(traceback.format_exc())  # never let cleanup stop the queue
            jobs.done()
            if not jobs.busy():
                emit({"type": "status", "msg": "Idle"})


if __name__ == "__main__":
    main()
