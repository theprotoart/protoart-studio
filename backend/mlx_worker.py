"""Fast engine: runs ERNIE-Image and FLUX.1 natively on the Apple GPU with MLX (via the open-source
mflux library), the same kind of engine Draw Things uses, instead of PyTorch.

Runs as its own process with its own Python environment (backend/.venv-mlx), because mflux
needs different library versions than the main engine. Talks JSON lines over stdin/stdout:

  in : {"cmd": "load", "repo": ..., "base": "ernie-image-turbo", "loras": [[path, scale], ...], "low_ram": bool}
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


class Worker:
    def __init__(self):
        self.model = None
        self.key = None
        self.low_ram = False
        self.family = "ernie"

    def load(self, repo, base="ernie-image-turbo", loras=None, low_ram=False, force=False, allow_download=False,
             label=None, size_gb=None):
        loras = [tuple(x) for x in (loras or [])]
        key = (repo, base, tuple(loras))
        if self.model is not None and self.key == key and not force:
            return
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        self.family = "flux" if base in FLUX_BASES else "ernie"
        name = label or ("FLUX" if self.family == "flux" else "ERNIE")

        self.model = None
        gc.collect()
        mx.clear_cache()
        path = repo
        if "/" in repo and not os.path.exists(repo):
            from huggingface_hub import snapshot_download
            try:
                path = snapshot_download(repo, local_files_only=True)
            except Exception:
                if not allow_download:
                    raise RuntimeError(f"offline: {name} isn't downloaded yet")
                import huggingface_hub.constants as hc
                hc.HF_HUB_OFFLINE = False
                os.environ["HF_HUB_OFFLINE"] = "0"
                send({"type": "status", "msg": f"Downloading {name} (first time only, ~{size_gb or '?'} GB)…"})
                try:
                    path = snapshot_download(repo)
                finally:
                    hc.HF_HUB_OFFLINE = True
                    os.environ["HF_HUB_OFFLINE"] = "1"
        send({"type": "status", "msg": f"Loading {name} (fast engine)…"})
        t0 = time.time()
        lora_kw = dict(lora_paths=[p for p, _ in loras] or None, lora_scales=[float(s) for _, s in loras] or None)
        if self.family == "flux":
            from mflux.models.flux.variants.txt2img.flux import Flux1
            config = {"schnell": ModelConfig.schnell, "dev": ModelConfig.dev, "krea-dev": ModelConfig.krea_dev}[base]()
            self.model = Flux1(model_path=path, model_config=config, **lora_kw)
        else:
            from mflux.models.ernie_image import ErnieImage
            config = ModelConfig.ernie_image_turbo() if base == "ernie-image-turbo" else ModelConfig.ernie_image()
            self.model = ErnieImage(model_path=path, model_config=config, **lora_kw)
        self.key = key
        self.low_ram = bool(low_ram)
        if self.low_ram:
            mx.set_cache_limit(1 << 30)
        self.model.callbacks.register(_Hooks(self))
        send({"type": "status", "msg": f"{name} ready ({time.time() - t0:.0f}s)"})
        print(f"[mlx] loaded {repo} in {time.time() - t0:.1f}s low_ram={self.low_ram}", file=sys.stderr, flush=True)

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
        else:
            cache_key = (a["prompt"], neg, cfg)
            missing = m.text_encoder is None
        if missing and cache_key not in m.prompt_cache:
            # Low-memory mode dropped the text encoder after the last prompt: reload to encode this one.
            repo, base, loras = self.key
            self.load(repo, base, list(loras), self.low_ram, force=True)
            m = self.model
        self.hooks_state = {"steps": int(a["steps"]), "preview": bool(a.get("preview")), "last": 0.0,
                            "started": time.time(), "width": int(a["width"]), "height": int(a["height"])}
        kw = dict(seed=int(a["seed"]), prompt=a["prompt"], num_inference_steps=int(a["steps"]),
                  width=int(a["width"]), height=int(a["height"]), guidance=cfg)
        if self.family != "flux":
            kw["negative_prompt"] = neg
        res = m.generate_image(**kw)
        res.image.save(a["out"])
        if len(m.prompt_cache) > 16:
            m.prompt_cache.clear()
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
            if getattr(self.w, "family", "ernie") == "flux":
                self.w.model.t5_text_encoder = None   # ~2-5 GB
                self.w.model.clip_text_encoder = None
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
        if st.get("preview") and done < steps and now - st.get("last", 0) > 1.5:
            try:
                import numpy as np
                import mlx.core as mx
                import preview
                lat = np.array(latents[0].astype(mx.float32))
                fam = getattr(self.w, "family", "ernie")
                if fam == "flux":
                    lat = preview.unpack_flux_tokens(lat, st.get("height", 1024), st.get("width", 1024))
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
            with jobs_cv:
                jobs.append(msg)
                jobs_cv.notify()
        with jobs_cv:
            jobs.append({"cmd": "quit"})
            jobs_cv.notify()

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
                            size_gb=msg.get("size_gb"))
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
