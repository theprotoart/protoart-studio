"""Fast engine: runs ERNIE-Image, FLUX.1 and FLUX.2 [klein] natively on the Apple GPU with MLX (via the open-source
mflux library), the same kind of engine Draw Things uses, instead of PyTorch.

Runs as its own process with its own Python environment (backend/.venv-mlx), because mflux
needs different library versions than the main engine. Talks JSON lines over stdin/stdout:

  in : {"cmd": "load", "repo": ..., "revision": None, "base": "ernie-image-turbo", "loras": [[path, scale], ...],
        "low_ram": bool}
       {"cmd": "generate", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed",
        "out": "/tmp/x.png", "preview": bool}
       {"cmd": "cancel"}        (stops the current image)
       {"cmd": "quit"}
  out: {"type": "status"|"progress"|"preview"|"done"|"error"|"loaded", ...}
"""
import gc
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_out_lock = threading.Lock()
cancel = threading.Event()


def send(msg):
    with _out_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


class Cancelled(Exception):
    pass


FLUX_BASES = ("schnell", "dev", "krea-dev")
FLUX2_BASES = ("flux2-klein-4b", "flux2-klein-9b", "flux2-klein-9b-kv", "flux2-klein-base-4b", "flux2-klein-base-9b")

# Every other model mflux can run is built and used the same way: its class, made with the model's config (found by
# its mflux name, e.g. "z-image-turbo"), a folder of weights and LoRAs, then generate_image(...). So a newer model is
# a catalog entry (its mflux name and, if mflux names its class differently, "module:Class"), not new code here.
GENERIC = {
    "zimage": "mflux.models.z_image.variants.z_image:ZImage",
    "qwen21": "mflux.models.qwen21.variants.txt2img.qwen_image_21:QwenImage21",
    "krea2": "mflux.models.krea2.variants.txt2img.krea2:Krea2",
    "fibo2": "mflux.models.fibo2.variants.txt2img.fibo2:Fibo2",
}
GENERIC_BASES = (("z-image", "zimage"), ("qwen-image-2.1", "qwen21"), ("krea-2", "krea2"), ("fibo2", "fibo2"))


def _family(base, cls=None):
    if cls:
        return "generic"
    if base in FLUX_BASES:
        return "flux"
    if base in FLUX2_BASES or str(base).startswith("flux2-klein"):
        return "flux2"
    for prefix, fam in GENERIC_BASES:
        if str(base).startswith(prefix):
            return fam
    return "ernie"


FAMILY_NAME = {"flux": "FLUX", "flux2": "FLUX.2", "ernie": "ERNIE", "zimage": "Z-Image", "qwen21": "Qwen-Image 2.1",
               "krea2": "Krea 2", "generic": "the model"}


def total_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def bake_loras():
    """Fold LoRAs into the model's weights only on Macs with room for the bigger result (24 GB or more), unless
    PROTOART_BAKE_LORA says otherwise ("1" always, "0" never)."""
    env = os.environ.get("PROTOART_BAKE_LORA")
    if env in ("0", "1"):
        return env == "1"
    return total_ram_gb() >= 24


def _make(cls, **kw):
    """Build an mflux model; an mflux without the bake_lora option gets the call without it."""
    try:
        return cls(**kw)
    except TypeError as e:
        if "bake_lora" in kw and "bake_lora" in str(e):
            kw.pop("bake_lora")
            return cls(**kw)
        raise


def _internal_copy(path, name):
    """Small Macs: use a copy on this Mac's own drive of a model kept on an external drive (once).

    The fast engine reads a model's weights from disk while the graphics chip waits for them. From a slow external
    drive (37 MB/s measured) the first step of a freshly downloaded model waited so long that macOS's GPU watchdog
    stopped it ("GPU Timeout Error"). On Macs with 14 GB or more, or without the free space, the original is used."""
    import shutil
    from pathlib import Path
    src = Path(path)
    try:
        real = str(src.resolve())
    except OSError:
        return path
    if not real.startswith("/Volumes/") or not 0 < total_ram_gb() < 14:
        return path
    home = Path(os.environ.get("PROTOART_HOME") or (Path.home() / ".protoartstudio"))
    dest = home / "fast-copies" / f"{src.parent.parent.name}--{src.name}"  # models--org--name--<revision>
    if (dest / ".protoart-done").exists():
        return str(dest)
    files = [f for f in src.rglob("*") if f.is_file()]
    need = sum(f.stat().st_size for f in files)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest.parent).free < need + (3 << 30):
        print(f"[mlx] not enough space to copy {name} off the external drive; reading it from there", file=sys.stderr,
              flush=True)
        return path
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    done = 0
    for f in files:
        out = part / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        send({"type": "status", "msg": f"Copying {name} from the external drive to this Mac (one time): "
                                       f"{done / 2**30:.1f} of {need / 2**30:.1f} GB…"})
        shutil.copyfile(f.resolve(), out)
        done += out.stat().st_size
    (part / ".protoart-done").write_text(real)
    shutil.rmtree(dest, ignore_errors=True)
    part.rename(dest)
    return str(dest)


class Worker:
    def __init__(self):
        self.model = None
        self.key = None
        self.low_ram = False
        self.family = "ernie"
        self.revision = None
        self.quantize = None
        self.patterns = None
        self.cls_path = None
        self.f2_cache = {}   # FLUX.2: encoded prompts, so the text encoder can be dropped between prompts
        self.zi_cache = {}   # Z-Image: the same
        self.path = None

    def load(self, repo, base="ernie-image-turbo", loras=None, low_ram=False, force=False, allow_download=False,
             label=None, size_gb=None, revision=None, variant=None, quantize=None, patterns=None, cls=None):
        loras = [tuple(x) for x in (loras or [])]
        variant = variant or None
        key = (repo, base, tuple(loras), variant)
        if revision is not None or not force:
            self.revision = revision
        if not force:
            self.quantize, self.patterns = quantize, patterns
        if self.model is not None and self.key == key and not force:
            return
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        self.family = _family(base, cls)
        self.cls_path = cls or GENERIC.get(self.family)
        name = label or FAMILY_NAME[self.family]
        self.f2_cache = {}
        self.zi_cache = {}

        self.model = None
        gc.collect()
        mx.clear_cache()
        path = repo
        if "/" in repo and not os.path.exists(repo):
            from huggingface_hub import snapshot_download
            try:
                path = snapshot_download(repo, local_files_only=True, revision=self.revision,
                                         allow_patterns=getattr(self, "patterns", None))
            except Exception:
                if not allow_download:
                    raise RuntimeError(f"offline: {name} isn't downloaded yet")
                import huggingface_hub.constants as hc
                hc.HF_HUB_OFFLINE = False
                os.environ["HF_HUB_OFFLINE"] = "0"
                send({"type": "status", "msg": f"Downloading {name} (first time only, ~{size_gb or '?'} GB)…"})
                try:
                    path = snapshot_download(repo, revision=self.revision, allow_patterns=getattr(self, "patterns", None))
                finally:
                    hc.HF_HUB_OFFLINE = True
                    os.environ["HF_HUB_OFFLINE"] = "1"
        path = _internal_copy(path, name)
        send({"type": "status", "msg": f"Loading {name} (fast engine)…"})
        t0 = time.time()
        lora_kw = dict(lora_paths=[p for p, _ in loras] or None, lora_scales=[float(s) for _, s in loras] or None)
        if loras and not bake_loras():
            # Keep LoRAs as small add-ons beside the 4-bit model instead of folding them in: folding re-saves every
            # touched layer at 8 bits (a 4-bit step is too coarse for the LoRA's change), which doubles the model's
            # size. On an 8 GB Mac that pushed ERNIE into swap (80–110 s a step instead of ~10–20). Unfolded, the
            # whole LoRA is applied too (folding kept only part of it), for a little extra work per step.
            lora_kw["bake_lora"] = False
        if self.family == "flux":
            from mflux.models.flux.variants.txt2img.flux import Flux1
            config = {"schnell": ModelConfig.schnell, "dev": ModelConfig.dev, "krea-dev": ModelConfig.krea_dev}[base]()
            q = getattr(self, "quantize", None)  # official full-size files: shrink while loading (FLUX.1 Dev)
            self.model = _make(Flux1, model_path=path, model_config=config, **({"quantize": int(q)} if q else {}), **lora_kw)
        elif self.family == "flux2":
            # "edit": the same weights, but it also looks at reference pictures (Image to image → Edit with words)
            from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
            cls = Flux2KleinEdit if variant == "edit" else Flux2Klein
            config = ModelConfig.flux2_klein_4b() if base == "flux2-klein-4b" else ModelConfig.from_name(base)
            self.model = _make(cls, model_path=path, model_config=config, **lora_kw)
            self._cache_flux2_prompts(self.model)
        elif self.cls_path:
            import importlib
            mod, _, name = self.cls_path.partition(":")
            if not (mod == "mflux" or mod.startswith("mflux.")) or not name.isidentifier():
                raise ValueError(f"{self.cls_path} isn't a fast-engine model class")
            klass = getattr(importlib.import_module(mod), name)
            q = getattr(self, "quantize", None)  # full-size files: shrink while loading
            self.model = _make(klass, model_path=path, model_config=ModelConfig.from_name(base),
                               **({"quantize": int(q)} if q else {}), **lora_kw)
            if self.family == "zimage":
                self._cache_zimage_prompts(self.model)
        else:
            from mflux.models.ernie_image import ErnieImage
            config = ModelConfig.ernie_image_turbo() if base == "ernie-image-turbo" else ModelConfig.ernie_image()
            self.model = _make(ErnieImage, model_path=path, model_config=config, **lora_kw)
        self.key = key
        self.path = path
        self.low_ram = bool(low_ram)
        if self.low_ram:
            mx.set_cache_limit(1 << 30)
        self.model.callbacks.register(_Hooks(self))
        send({"type": "status", "msg": f"{name} ready ({time.time() - t0:.0f}s)"})
        print(f"[mlx] loaded {repo} in {time.time() - t0:.1f}s low_ram={self.low_ram}", file=sys.stderr, flush=True)

    def _cache_flux2_prompts(self, m):
        """FLUX.2 encodes the prompt on every picture and has no prompt cache of its own: keep the encoded prompts
        here, so on small Macs the ~2 GB text encoder can be dropped while the picture is made (see _Hooks)."""
        orig = m._encode_prompt_pair
        cache = self.f2_cache

        def encode(*, prompt, negative_prompt, guidance):
            k = (prompt, bool(guidance is not None and guidance > 1.0))
            if k not in cache:
                if len(cache) > 16:
                    cache.clear()
                cache[k] = orig(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
            return cache[k]

        m._encode_prompt_pair = encode   # a plain function: MLX keeps it as an attribute, not a weight

    def _cache_zimage_prompts(self, m):
        """Z-Image: keep encoded prompts, so on small Macs its Qwen3 text encoder (~2.2 GB at 4-bit) can be dropped
        while the picture is made and the 6B picture model has the memory to itself (see _Hooks)."""
        orig = m._encode_prompts
        cache = self.zi_cache

        def encode(*, prompt, negative_prompt, guidance):
            k = (prompt, (negative_prompt or "") if guidance and guidance > 1.0 else "", bool(guidance and guidance > 1.0))
            if k not in cache:
                if len(cache) > 16:
                    cache.clear()
                cache[k] = orig(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
            return cache[k]

        m._encode_prompts = encode

    def _reload_zimage_text_encoder(self):
        """Bring back only Z-Image's text encoder for a new prompt (instead of reloading the whole model)."""
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
        weights = WeightLoader.load(weight_definition=ZImageWeightDefinition, model_path=self.path)
        comp = next(c for c in ZImageWeightDefinition.get_components() if c.name == "text_encoder")
        te = TextEncoder()
        WeightApplier.apply_and_quantize_single(weights=weights, model=te, component=comp, quantize_arg=None)
        del weights
        self.model.text_encoder = te

    def generate(self, a):
        import mlx.core as mx
        m = self.model
        if m is None:
            raise RuntimeError("model not loaded")
        cfg = float(a.get("cfg") or 1.0)
        neg = a.get("negative_prompt") or None
        if self.family == "flux":
            neg = None  # FLUX Schnell has no negative prompt
            cache_key = a["prompt"]
            missing = getattr(m, "t5_text_encoder", None) is None
        elif self.family == "flux2":
            neg = None  # FLUX.2 [klein] is distilled: no negative prompt, guidance stays at 1
            cache_key = (a["prompt"], cfg > 1.0)
            missing = getattr(m, "text_encoder", None) is None
        elif self.family == "zimage":
            g = cfg if cfg > 1.0 and getattr(getattr(m, "model_config", None), "supports_guidance", False) else 0.0
            cache_key = (a["prompt"], (neg or "") if g > 1.0 else "", g > 1.0)
            missing = getattr(m, "text_encoder", None) is None
        elif self.cls_path:
            cache_key, missing = None, False  # these keep their text encoder (see _Hooks)
        else:
            cache_key = (a["prompt"], neg, cfg)
            missing = m.text_encoder is None
        prompt_cache = self.f2_cache if self.family == "flux2" else self.zi_cache if self.family == "zimage" \
            else getattr(m, "prompt_cache", {})
        if missing and cache_key not in prompt_cache and self.family == "zimage":
            try:
                send({"type": "status", "msg": "Loading Z-Image's text reader for the new prompt…"})
                self._reload_zimage_text_encoder()
                missing = False
            except Exception as e:  # mflux changed inside: reload the whole model instead
                print(f"[mlx] Z-Image text encoder reload failed ({e}); reloading the model", file=sys.stderr, flush=True)
        if missing and cache_key not in prompt_cache:
            # Low-memory mode dropped the text encoder after the last prompt: reload to encode this one.
            repo, base, loras, variant = self.key
            self.load(repo, base, list(loras), self.low_ram, force=True, revision=self.revision, variant=variant)
            m = self.model
        self.hooks_state = {"steps": int(a["steps"]), "preview": bool(a.get("preview")), "last": 0.0,
                            "started": time.time(), "width": int(a["width"]), "height": int(a["height"])}
        kw = dict(seed=int(a["seed"]), prompt=a["prompt"], num_inference_steps=int(a["steps"]),
                  width=int(a["width"]), height=int(a["height"]), guidance=cfg)
        if self.family == "ernie":
            kw["negative_prompt"] = neg
        elif self.cls_path:
            import inspect
            params = inspect.signature(m.generate_image).parameters
            if neg and cfg > 1.0 and "negative_prompt" in params:
                kw["negative_prompt"] = neg
        if a.get("init_image"):  # start from a picture: how much of it to keep (mflux image strength)
            import inspect
            if "image_path" not in inspect.signature(m.generate_image).parameters:
                raise RuntimeError("this model can't start from a picture on the fast engine")
            kw["image_path"] = a["init_image"]
            kw["image_strength"] = float(a.get("image_strength") or 0.4)
        if a.get("image_paths"):
            if self.family != "flux2" or (self.key and self.key[3] != "edit"):
                raise RuntimeError("reference pictures need FLUX.2 loaded for editing")
            kw["image_paths"] = list(a["image_paths"])
        res = m.generate_image(**kw)
        res.image.save(a["out"])
        pc = getattr(m, "prompt_cache", None)
        if self.family != "flux2" and isinstance(pc, dict) and len(pc) > 16:
            pc.clear()
        gc.collect()
        mx.clear_cache()
        return a["out"]


class _Hooks:
    """mflux callbacks: progress, live preview, Stop, and freeing the text encoder on small Macs."""

    def __init__(self, worker):
        self.w = worker

    def call_before_loop(self, seed, prompt, latents, config, **kw):
        if self.w.low_ram and self.w.model is not None:
            import mlx.core as mx
            # The prompt is encoded and cached: drop the text encoder(s) while the picture is made, so the
            # image model has the memory to itself (on 8 GB Macs this avoids swapping, the main slowdown).
            fam = getattr(self.w, "family", "ernie")
            if fam == "zimage":
                self.w.model.text_encoder = None      # Qwen3-4B, ~2.2 GB at 4-bit (the prompt is cached)
                gc.collect()
                mx.clear_cache()
                return
            if getattr(self.w, "cls_path", None):
                return  # newer models: their text encoder stays (each is built differently; mflux frees what it can)
            if fam == "flux":
                self.w.model.t5_text_encoder = None   # ~2-5 GB
                self.w.model.clip_text_encoder = None
            elif fam == "flux2":
                self.w.model.text_encoder = None      # Qwen3, ~2 GB at 4-bit
            else:
                self.w.model.text_encoder = None  # ~2 GB
            gc.collect()
            mx.clear_cache()

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        st = getattr(self.w, "hooks_state", {})
        if cancel.is_set():
            raise Cancelled()
        steps = st.get("steps") or config.num_inference_steps
        done = (t - getattr(config, "init_time_step", 0)) + 1
        send({"type": "progress", "step": min(done, steps), "steps": steps})
        now = time.time()
        if st.get("preview") and done < steps and now - st.get("last", 0) > 1.5 and not getattr(self.w, "cls_path", None):
            try:
                import numpy as np
                import mlx.core as mx
                import preview
                lat = np.array(latents[0].astype(mx.float32))
                fam = getattr(self.w, "family", "ernie")
                if fam == "flux":
                    lat = preview.unpack_flux_tokens(lat, st.get("height", 1024), st.get("width", 1024))
                elif fam == "flux2":
                    lat = preview.unpack_flux2_tokens(lat, st.get("height", 1024), st.get("width", 1024))
                send({"type": "preview", "src": preview.latent_to_data_url(lat, fam)})
                st["last"] = now
            except Exception as e:
                print("[mlx] preview failed:", e, file=sys.stderr, flush=True)

    def call_after_loop(self, seed, prompt, latents, config):
        pass


def main():
    worker = Worker()
    jobs = []
    jobs_cv = threading.Condition()

    def reader():
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("cmd") == "cancel":
                cancel.set()
                continue
            if msg.get("cmd") == "quit":
                _exit_soon()
            with jobs_cv:
                jobs.append(msg)
                jobs_cv.notify()
        # The app closed (or crashed): finish up and stop, even in the middle of a long step. A worker left running
        # on its own keeps gigabytes of memory, and on 8 GB Macs that starved everything after it ("GPU Timeout Error").
        with jobs_cv:
            jobs.append({"cmd": "quit"})
            jobs_cv.notify()
        _exit_soon(stop_work=False)

    def _exit_soon(stop_work=True):
        if stop_work:
            cancel.set()
        t = threading.Timer(5.0, lambda: os._exit(0))  # a quit the busy main loop can't get to in time
        t.daemon = True
        t.start()

    threading.Thread(target=reader, daemon=True).start()
    send({"type": "ready"})
    while True:
        with jobs_cv:
            while not jobs:
                jobs_cv.wait()
            msg = jobs.pop(0)
        cmd = msg.get("cmd")
        if cmd == "quit":
            break
        cancel.clear()
        try:
            if cmd == "load":
                worker.load(msg["repo"], msg.get("base", "ernie-image-turbo"), msg.get("loras"), msg.get("low_ram"),
                            allow_download=bool(msg.get("allow_download")), label=msg.get("label"),
                            size_gb=msg.get("size_gb"), revision=msg.get("revision"), variant=msg.get("variant"),
                            quantize=msg.get("quantize"), patterns=msg.get("patterns"), cls=msg.get("cls"))
                send({"type": "loaded"})
            elif cmd == "generate":
                send({"type": "done", "path": worker.generate(msg)})
        except Cancelled:
            send({"type": "error", "cancelled": True, "msg": "Stopped"})
        except Exception as e:
            traceback.print_exc()
            send({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
