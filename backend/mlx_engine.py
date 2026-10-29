"""Manages the fast ERNIE engine (MLX): a one-time install into backend/.venv-mlx and a worker
process (mlx_worker.py) that keeps the model loaded between images."""
import json
import signal
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_OLD_VENV = HERE / ".venv-mlx"
# Installed into the user's data folder, not next to the app: inside a packaged .app the app's own
# folder is read-only. An existing install beside the code (from earlier versions) keeps being used.
_HOME = Path(os.environ.get("PROTOART_HOME") or (Path.home() / ".protoartstudio"))
VENV = _OLD_VENV if (_OLD_VENV / ".protoart-installed").exists() else _HOME / "venv-mlx"
PY = VENV / "bin" / "python"
MARKER = VENV / ".protoart-installed"
MFLUX_SPEC = "mflux==0.22.0"  # exact tested version
INSTALL_GB = 1.5


class EngineError(Exception):
    pass


def friendly(msg):
    """Turn the fast engine's technical errors into something a person can act on."""
    import re
    m = re.search(r"layers named by (.+?) could not be applied", msg or "")
    if m:
        return (f"The LoRA “{m.group(1).rsplit('.', 1)[0]}” doesn't fit this model (it was made for a different model "
                "type, e.g. FLUX.1 instead of FLUX.2). Remove it from the LoRA list, or check its type in "
                "Models → Fix a type.")
    return msg


class CancelledError(Exception):
    pass


def supported():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def installed():
    # PY is a link to the Python the environment was made with; if that Python was removed, reinstall
    return PY.exists() and MARKER.exists() and os.path.exists(os.path.realpath(PY))


class MlxEngine:
    def __init__(self, emit, cancel_event, pause_event=None):
        self.emit = emit
        self.cancel_event = cancel_event
        self.pause_event = pause_event if pause_event is not None else threading.Event()
        self.frozen = False  # the worker is suspended (Pause)
        self.proc = None
        self.q = queue.Queue()
        self.err_tail = ""
        self.python = PY                          # a subclass (MOSS-Image) runs its own worker in its own Python
        self.worker_script = HERE / "mlx_worker.py"

    # ------------------------------------------------------------- install
    def install(self):
        if not supported():
            raise EngineError("The fast engine needs a Mac with Apple Silicon (M1 or newer).")
        if installed():
            return
        VENV.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(VENV.parent).free / 2**30
        if free < INSTALL_GB + 1:
            raise EngineError(f"Installing the fast engine needs about {INSTALL_GB} GB free (plus the model); "
                              f"only {free:.1f} GB is free.")
        self.emit({"type": "status", "msg": "Installing the fast engine (one time, a few minutes)…"})
        if VENV.exists() and not installed():  # broken or half-finished install: start over
            shutil.rmtree(VENV, ignore_errors=True)
        if not PY.exists():
            base = getattr(sys, "_base_executable", None) or sys.executable
            self._run([base, "-m", "venv", str(VENV)])
        self._run([str(PY), "-m", "pip", "install", "--upgrade", "pip"], quiet=True)
        self._run([str(PY), "-m", "pip", "install", MFLUX_SPEC])
        MARKER.write_text(MFLUX_SPEC)
        self.emit({"type": "status", "msg": "Fast engine installed"})

    def _run(self, cmd, quiet=False):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(HERE))
        tail = []
        last = 0
        for line in p.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-30:]
            print("[mlx-install]", line, file=sys.stderr, flush=True)
            if self.cancel_event.is_set():
                p.kill()
                raise CancelledError()
            if not quiet and line.startswith(("Collecting", "Downloading", "Installing")) and time.time() - last > 1:
                self.emit({"type": "status", "msg": "Installing the fast engine: " + line[:90]})
                last = time.time()
        if p.wait() != 0:
            raise EngineError("Installing the fast engine failed:\n" + "\n".join(tail[-8:]))

    # ------------------------------------------------------------- worker process
    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        self.q = queue.Queue()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen([str(self.python), str(self.worker_script)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     cwd=str(HERE), env=env, bufsize=1)
        proc = self.proc

        def read_out():
            for line in proc.stdout:
                try:
                    self.q.put(json.loads(line))
                except ValueError:
                    print("[mlx]", line.rstrip(), file=sys.stderr, flush=True)
            self.q.put({"type": "exit"})

        def read_err():
            for line in proc.stderr:
                self.err_tail = (self.err_tail + line)[-3000:]
                print("[mlx]", line.rstrip(), file=sys.stderr, flush=True)

        threading.Thread(target=read_out, daemon=True).start()
        threading.Thread(target=read_err, daemon=True).start()
        try:
            msg = self.q.get(timeout=180)
        except queue.Empty:
            msg = {"type": "timeout"}
        if msg.get("type") != "ready":
            self.stop()
            if proc.poll() is None:
                proc.kill()
            raise EngineError("The fast engine didn't start:\n" + self.err_tail[-800:])

    def _freeze(self, on):
        """Pause / resume the fast engine's worker process (it keeps its memory; nothing is lost)."""
        p = self.proc
        if not p or p.poll() is not None:
            self.frozen = False
            return
        if on == self.frozen:
            return
        try:
            os.kill(p.pid, signal.SIGSTOP if on else signal.SIGCONT)
            self.frozen = on
        except OSError:
            self.frozen = False

    def stop(self):
        self._freeze(False)
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            try:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def request(self, msg, on_event=None, finish=("done", "loaded"), skip_event=None):
        """Send one command and wait for its result, passing progress/preview/status events on."""
        self.start()
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        try:
            return self._request_loop(on_event, finish, skip_event)
        finally:
            self._freeze(False)

    def _request_loop(self, on_event, finish, skip_event):
        sent_cancel = False
        last_cancel = 0.0
        while True:
            stopping = self.cancel_event.is_set() or (skip_event is not None and skip_event.is_set())
            # Pause: suspend the worker; Resume / Stop / Skip wake it again
            self._freeze(self.pause_event.is_set() and not stopping)
            # Repeat the stop request every 2 s: one sent just before the worker picked up the job
            # would otherwise be cleared when the job starts, and the image would run to the end.
            if (self.cancel_event.is_set() or (skip_event is not None and skip_event.is_set())) \
                    and time.time() - last_cancel > 2:
                try:
                    self.proc.stdin.write(json.dumps({"cmd": "cancel"}) + "\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
                sent_cancel = True
                last_cancel = time.time()
            try:
                ev = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            t = ev.get("type")
            if t in finish:
                return ev
            if t == "error":
                if ev.get("cancelled"):
                    raise CancelledError()
                raise EngineError(friendly(ev.get("msg")) or "fast engine error")
            if t == "exit":
                self.proc = None
                if sent_cancel:
                    raise CancelledError()
                raise EngineError("The fast engine stopped unexpectedly (often: not enough memory).\n"
                                  + self.err_tail[-600:])
            if on_event:
                on_event(ev)
