"""MOSS-Image worker: runs in its own Python (~/.protoartstudio/venv-moss) with moss-image-mlx. JSON lines in and out,
like the fast engine's worker:

  in : {"cmd": "load", "dir": <weights folder>, "turbo": bool, "adapters": [student, destripe], "cache": <folder>,
        "student_bf16": bool}
       {"cmd": "shrink", "src": <weights folder>, "out": <folder>}   (8 GB Macs, once: see moss_small.py)
       {"cmd": "generate", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed", "out": "/tmp/x.png"}
       {"cmd": "cancel"}  {"cmd": "quit"}
  out: {"type": "ready"|"status"|"progress"|"done"|"loaded"|"error", ...}

The pipeline loads each part when it's needed and frees it after (text encoder, then transformer, then the picture
decoder), so the most memory in use at once is about one part's worth. Encoded prompts are kept on disk, so making
another picture of the same prompt skips the 9B text encoder.
"""
import json
import os
import sys
import threading
import time
import traceback

_out_lock = threading.Lock()
cancel = threading.Event()


def send(msg):
    with _out_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


class Cancelled(Exception):
    pass


class _EvalEach:
    """Wraps one text-encoder layer for streaming (see generate): run it, finish its GPU work, then let its weights go.
    The encoder's weights are loaded lazily from disk, so only the layer being run is in memory."""

    def __init__(self, layer, i, n):
        self._layer, self._i, self._n = layer, i, n
        self.is_linear = getattr(layer, "is_linear", False)  # read by the model before each layer runs

    def __call__(self, *args, **kwargs):
        import mlx.core as mx
        if cancel.is_set():
            raise Cancelled()
        out = self._layer(*args, **kwargs)
        mx.eval(out)
        self._layer = None  # done with this layer: its weights leave memory
        mx.clear_cache()
        if self._i % 4 == 0:
            send({"type": "progress", "step": self._i + 1, "steps": self._n})
        return out

    def __getattr__(self, name):
        layer = self.__dict__.get("_layer")
        if layer is None:
            raise AttributeError(name)
        return getattr(layer, name)


class Worker:
    def __init__(self):
        self.pipe = None
        self.key = None

    def load(self, a):
        user = [(str(p), float(w)) for p, w in a.get("loras") or []]  # your LoRAs: [path, strength]
        key = (a["dir"], bool(a.get("turbo")), tuple(a.get("adapters") or ()), bool(a.get("student_bf16")), tuple(user))
        if self.pipe is not None and self.key == key:
            return
        import mlx.core as mx
        from moss_image_mlx import MOSSImagePipeline
        self.pipe = None
        loras = []
        if a.get("turbo"):
            student, destripe = a["adapters"]
            # destripe first (always on), then the student wrapped around it (switched per step), as the port does
            # the student is kept in float32 for precision; on small Macs bf16 halves its ~0.5 GB
            loras = [(destripe, float(a.get("destripe_strength", 1.5)), mx.bfloat16, "destripe"),
                     (student, 1.0, mx.bfloat16 if a.get("student_bf16") else mx.float32, "student")]
        if user:
            # your LoRAs go on after the app's own adapters, in their own group so the turbo schedule (which switches
            # the "student" group per step) and per-step strengths ("default") leave them at the strength you set;
            # moss_lora reads the LoRA / PEFT / LoKr formats moss-image-mlx doesn't
            import moss_image_mlx.lora as mlora
            import moss_lora
            if not getattr(mlora, "_protoart_any", False):
                plain = mlora.apply_lora

                def apply_any(model, path, strength=1.0, dtype=mx.bfloat16, group="default"):
                    if group == "user":
                        return moss_lora.apply(model, path, strength, dtype, group)
                    return plain(model, path, strength, dtype, group)

                mlora.apply_lora = apply_any
                mlora._protoart_any = True
            loras += [(p, w, mx.bfloat16, "user") for p, w in user]
        self.pipe = MOSSImagePipeline(a["dir"], low_memory=True, cache_dir=a.get("cache") or None, loras=loras)
        self.key = key
        self.cpu_encode = bool(a.get("cpu_encode"))

    def shrink(self, a):
        """Make the small copy for 8 GB Macs: the picture model re-stored with 4-bit feed-forward layers (8-bit
        attention and end blocks, see moss_small.py). One layer at a time on the CPU, so the 8-bit original is never
        in memory whole; the text encoder, tokenizer, scheduler and picture decoder are linked, not copied."""
        import gc
        import shutil
        from pathlib import Path

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from moss_image_mlx.dit import load_transformer
        from moss_image_mlx.export import _save_sharded
        from moss_image_mlx.quantize import DIT_SKIP

        from moss_small import bits_for, small_spec

        self.pipe, self.key = None, None
        gc.collect()
        mx.clear_cache()
        src, out = Path(a["src"]), Path(a["out"])
        part = out.with_name(out.name + ".part")
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        cfg = json.loads((src / "transformer" / "config.json").read_text())
        old = (cfg.get("moss_image_mlx") or {}).get("quantization")
        if not old or old.get("mode", "affine") != "affine":
            raise RuntimeError("this MOSS-Image download isn't the expected 8-bit format, so it can't be made smaller")
        started = time.time()
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = load_transformer(src / "transformer")
            spec = small_spec(old["group_size"], old.get("skip", DIT_SKIP), len(model.layers))
            todo = [(p, m) for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)]
            checked = False
            for i, (path, m) in enumerate(todo):
                if cancel.is_set():
                    raise Cancelled()
                bits = bits_for(spec, path)
                if bits != m.bits:
                    w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
                    wq, scales, biases = mx.quantize(w, group_size=spec["group_size"], bits=bits)
                    mx.eval(wq, scales, biases)
                    if not checked:  # the first converted layer must come back close to the original
                        back = mx.dequantize(wq, scales, biases, group_size=spec["group_size"], bits=bits)
                        err = (mx.linalg.norm((back - w).astype(mx.float32)) /
                               (mx.linalg.norm(w.astype(mx.float32)) + 1e-12)).item()
                        if not err < 0.25:
                            raise RuntimeError(f"self-check failed: layer {path} changed too much ({err:.2f})")
                        print(f"[moss] small copy self-check {path}: {err:.3f} relative error", file=sys.stderr,
                              flush=True)
                        checked = True
                    m.weight, m.scales, m.biases, m.bits, m.group_size = wq, scales, biases, bits, spec["group_size"]
                    del w
                else:
                    mx.eval(m.weight, m.scales, m.biases)
                if i % 8 == 0 or i == len(todo) - 1:
                    send({"type": "progress", "step": i + 1, "steps": len(todo), "elapsed": round(time.time() - started, 1)})
            params = dict(tree_flatten(model.parameters()))
            mx.eval(list(params.values()))
            size = sum(v.nbytes for v in params.values())
            _save_sharded(params, part / "transformer")
            del params, model
            gc.collect()
            mx.clear_cache()
        finally:
            mx.set_default_device(previous)
        cfg["moss_image_mlx"] = {**cfg["moss_image_mlx"], "quantization": spec}
        (part / "transformer" / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        for f in src.iterdir():  # everything else is shared with the download
            if f.name != "transformer":
                os.symlink(f.resolve() if f.is_file() else f, part / f.name)
        (part / ".protoart-done").write_text(json.dumps({"source": str(src), "transformer_gb": round(size / 2**30, 2),
                                                          "spec": spec}))
        shutil.rmtree(out, ignore_errors=True)
        part.rename(out)
        print(f"[moss] small copy made in {time.time() - started:.0f}s: transformer {size / 2**30:.2f} GB", file=sys.stderr,
              flush=True)
        return str(out)

    def generate(self, a):
        from PIL import Image
        from moss_image_mlx import Request
        if self.pipe is None:
            raise RuntimeError("model not loaded")
        turbo = self.key[1]
        req = Request(prompt=a["prompt"], negative_prompt=a.get("negative_prompt") or "",
                      width=int(a["width"]), height=int(a["height"]), steps=int(a.get("steps") or 50),
                      guidance=float(a.get("cfg") or 3.0), seed=int(a["seed"]) % 2**31, turbo=turbo)
        started = time.time()
        send({"type": "status", "msg": "Reading the prompt (MOSS-Image's text encoder)…"})

        def on_step(i, n):
            if cancel.is_set():
                raise Cancelled()
            if i == 0:
                send({"type": "status", "msg": "Drawing the picture…"})
            send({"type": "progress", "step": i + 1, "steps": n, "elapsed": round(time.time() - started, 1)})

        if getattr(self, "cpu_encode", False):
            # 8 GB Macs: the 9B text encoder (~7 GB) doesn't fit beside the system: held whole, it swaps, the GPU
            # stalls and macOS's GPU watchdog stops it ("GPU Timeout Error"); on the CPU it thrashes for many minutes.
            # So it's streamed: its weights load lazily from disk, and each layer is run on the GPU and dropped before
            # the next (~1 GB in memory instead of ~7). The result is kept on disk, so this is once per prompt.
            texts = [req.prompt] + ([req.negative_prompt] if req.uses_cfg() else [])
            cached = self.pipe.cache is not None and all(self.pipe.cache.get(t, req.max_sequence_length) is not None
                                                         for t in texts)
            if not cached:
                send({"type": "status", "msg": "Reading the prompt (8 GB Macs: once per prompt)…"})
                for t in texts:  # each prompt streams the encoder from disk once (its layers are dropped as they run)
                    if self.pipe.cache.get(t, req.max_sequence_length) is not None:
                        continue
                    self.pipe.text_encoder = None
                    self.pipe._load_text_encoder()
                    model = self.pipe.text_encoder.model
                    n = len(model.layers)
                    model.layers = [_EvalEach(layer, i, n) for i, layer in enumerate(model.layers)]
                    self.pipe.encode([t], req.max_sequence_length, keep=True)  # saved to the prompt cache
                    self.pipe.free_text_encoder()
                    if cancel.is_set():
                        raise Cancelled()
        image = self.pipe(req, on_step=on_step)
        if cancel.is_set():
            raise Cancelled()
        Image.fromarray(image).save(a["out"])
        print(f"[moss] {req.width}x{req.height} turbo={turbo} in {time.time() - started:.1f}s "
              f"{json.dumps(self.pipe.timings)}", file=sys.stderr, flush=True)
        return a["out"]


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the weights are downloaded by the app beforehand
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
                worker.load(msg)
                send({"type": "loaded"})
            elif cmd == "shrink":
                send({"type": "done", "path": worker.shrink(msg)})
            elif cmd == "generate":
                send({"type": "done", "path": worker.generate(msg)})
        except Cancelled:
            send({"type": "error", "cancelled": True, "msg": "Stopped"})
        except Exception as e:
            traceback.print_exc()
            send({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
