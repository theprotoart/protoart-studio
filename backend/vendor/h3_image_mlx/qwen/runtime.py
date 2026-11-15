"""Native worker execution, independent of model semantics and desktop state."""

import importlib.resources
import json
import os
import platform
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path


class Cancelled(RuntimeError):
    pass


def locate(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path = path / "qwen-stream-probe" if path.is_dir() else path
    else:
        try:
            path = Path(
                str(
                    importlib.resources.files("h3_image_mlx_native").joinpath(
                        "payload/qwen-stream-probe"
                    )
                )
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen requires the optional h3-image-mlx-qwen-runtime package, or QwenOptions(runtime=...). See docs/QWEN.md."
            ) from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Qwen runtime is not executable: {path}")
    manifest = path.parent / "runtime.json"
    if manifest.is_file():
        minimum = json.loads(manifest.read_text()).get("minimum_macos")
        if minimum:
            required = tuple(map(int, minimum.split(".")))
            current = tuple(map(int, platform.mac_ver()[0].split(".")[: len(required)]))
            if current < required:
                raise RuntimeError(
                    f"This Qwen runtime requires macOS {minimum} or later"
                )
    return path


def run_worker(
    command, *, input_bytes=None, timeout=900, cancel=None, on_progress=None, env=None
):
    """Drain all pipes concurrently and reap our process group on every exit path."""
    if cancel is not None and cancel.is_set():
        raise Cancelled("Generation cancelled")
    started = time.monotonic()
    p = subprocess.Popen(
        list(map(str, command)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
    )
    messages = queue.Queue()
    output = []
    errors = []

    def feed():
        try:
            if input_bytes:
                p.stdin.write(input_bytes)
                p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            p.stdin.close()

    def stdout():
        try:
            output.append(p.stdout.read())
        except Exception as exc:
            errors.append(exc)

    def stderr():
        try:
            for raw in p.stderr:
                messages.put(raw.decode("utf-8", errors="replace").rstrip())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=f, daemon=True) for f in (feed, stdout, stderr)]
    for thread in threads:
        thread.start()
    metrics = {}
    tail = []
    progress = []

    def consume(line):
        tail.append(line)
        if len(tail) > 12:
            del tail[0]
        event = None
        if line.startswith("PROGRESS "):
            event = json.loads(line[9:])
        match = re.match(r"step (\d+)/(\d+) (\d+) ms", line)
        if match:
            event = dict(stage="Denoising", step=int(match[1]), total=int(match[2]))
        if event:
            progress.append(event)
            if on_progress:
                on_progress(event)
        if line.startswith("METRICS "):
            metrics.update(json.loads(line[8:]))
        match = re.match(r"FOOTPRINT (\d+) DISK_READ (\d+)", line)
        if match:
            metrics.update(peak_bytes=int(match[1]), disk_read_bytes=int(match[2]))
        match = re.match(
            r"STRICT_IO calls=(\d+) logical_bytes=(\d+) aligned_bytes=(\d+) errors=(\d+)",
            line,
        )
        if match:
            metrics["weight_io"] = dict(
                zip(
                    ("calls", "logical_bytes", "aligned_bytes", "errors"),
                    map(int, match.groups()),
                )
            )

    try:
        while (
            p.poll() is None
            or any(t.is_alive() for t in threads)
            or not messages.empty()
        ):
            if cancel is not None and cancel.is_set():
                raise Cancelled("Generation cancelled")
            if time.monotonic() - started > timeout:
                raise TimeoutError("Qwen worker timed out")
            try:
                consume(messages.get(timeout=0.05))
            except queue.Empty:
                pass
        if errors:
            raise RuntimeError(f"Qwen pipe failure: {errors[0]}")
        if p.returncode:
            raise RuntimeError(
                f"Qwen worker exited {p.returncode}: " + "; ".join(tail)[-1800:]
            )
        metrics.update(wall_seconds=time.monotonic() - started, progress=progress)
        return b"".join(output), metrics
    finally:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for thread in threads:
            thread.join(timeout=5)
        p.stdout.close()
        p.stderr.close()
