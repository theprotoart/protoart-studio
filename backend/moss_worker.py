"""MOSS-Image worker: runs in its own Python (~/.protoartstudio/venv-moss) with moss-image-mlx. JSON lines in and out,
like the fast engine's worker:

  in : {"cmd": "load", "dir": <weights folder>, "turbo": bool, "adapters": [student, destripe], "cache": <folder>}
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


class Worker:
    def __init__(self):
        self.pipe = None
        self.key = None

    def load(self, a):
        key = (a["dir"], bool(a.get("turbo")), tuple(a.get("adapters") or ()))
        if self.pipe is not None and self.key == key:
            return
        import mlx.core as mx
        from moss_image_mlx import MOSSImagePipeline
        self.pipe = None
        loras = []
        if a.get("turbo"):
            student, destripe = a["adapters"]
            # destripe first (always on), then the student wrapped around it (switched per step), as the port does
            loras = [(destripe, float(a.get("destripe_strength", 1.5)), mx.bfloat16, "destripe"),
                     (student, 1.0, mx.float32, "student")]
        self.pipe = MOSSImagePipeline(a["dir"], low_memory=True, cache_dir=a.get("cache") or None, loras=loras)
        self.key = key

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
                worker.load(msg)
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
