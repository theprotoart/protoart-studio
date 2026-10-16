"""Stand-in for engine.Engine used by the queue tests: each 'image' takes 10 steps of 0.05 s."""
import threading
import time

from engine import Cancelled


class FakeEngine:
    def __init__(self, library, emit):
        self._emit = emit
        self.cancel_event = threading.Event()
        self.skip_event = threading.Event()
        self.pause_event = threading.Event()
        self.current_job = None

    def wait_if_paused(self):
        while self.pause_event.is_set() and not self.cancel_event.is_set():
            time.sleep(0.02)

    def release_memory(self):
        pass

    def emit(self, msg):
        self._emit(dict(msg, job=self.current_job))

    def generate(self, args, force=False, low_memory=False):
        for step in range(10):
            self.wait_if_paused()
            if self.cancel_event.is_set():
                raise Cancelled()
            time.sleep(0.05)
            self.emit({"type": "progress", "step": step + 1, "steps": 10, "image": 1, "images": 1})
        return [{"prompt": args.get("prompt")}]

    def preload(self, args, force=False, low_memory=False):
        for _ in range(10):  # "loading" takes 0.5 s
            if self.cancel_event.is_set():
                raise Cancelled()
            time.sleep(0.05)
        return {"ready": args.get("model")}
