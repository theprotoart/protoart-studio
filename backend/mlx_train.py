"""ERNIE LoRA training on the fast engine (MLX, via mflux's trainer).

The PyTorch trainer needs the full-precision ERNIE model and 32 GB+. mflux can train a LoRA on top of the
4-bit fast-engine model instead (QLoRA-style: the base stays compressed, only the small LoRA is learned),
with low-RAM options, so it can run on much smaller Macs. It runs as its own process in the fast
engine's Python (backend/.venv-mlx); this module prepares the data, writes the config, follows the
progress and copies the finished LoRA into the LoRA folder.
"""
import json
import math
import os
import threading
import signal
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ERNIE_TARGETS_ATTENTION = ("self_attention.to_q", "self_attention.to_k", "self_attention.to_v", "self_attention.to_out.0")
ERNIE_TARGETS_FULL = ERNIE_TARGETS_ATTENTION + ("mlp.gate_proj", "mlp.up_proj", "mlp.linear_fc2")
ERNIE_BLOCKS = 36

_PROGRESS = re.compile(r"(\d+)/(\d+) \[[^\]]*?(?:([\d.]+)s/it|([\d.]+)it/s)")


class TrainError(Exception):
    pass


class Cancelled(Exception):
    pass


def build_config(*, model_path, data_dir, out_dir, steps, n_samples, lr, rank, res, targets, save_every, ram_gb, seed=42):
    """mflux training config (same layout as its train_ernie_image_turbo.json example)."""
    epochs = max(1, math.ceil(steps / max(1, n_samples)))
    names = ERNIE_TARGETS_FULL if targets == "full" else ERNIE_TARGETS_ATTENTION
    lora = [{"module_path": f"layers.{{block}}.{n}", "blocks": {"start": 0, "end": ERNIE_BLOCKS}, "rank": int(rank)}
            for n in names]
    if targets == "full":
        lora.append({"module_path": "text_proj", "rank": int(rank)})
    small = 0 < ram_gb < 32
    cfg = {
        "model": "ernie-image-turbo",
        "model_path": str(model_path),
        "data": str(data_dir),
        "seed": int(seed),
        "steps": 8, "guidance": 1.0,   # the Turbo model's own sampling settings
        "quantize": None,               # the fast-engine model is already 4-bit
        "max_resolution": int(res),
        "low_ram": small,               # encoded data kept on disk instead of in memory
        "gradient_checkpointing": 0 < ram_gb < 64,  # ~30% slower, much less memory
        "training_loop": {"num_epochs": epochs, "batch_size": 1, "timestep_low": 1, "timestep_high": 8},
        "optimizer": {"name": "AdamW", "learning_rate": float(lr)},
        "checkpoint": {"output_path": str(out_dir), "save_frequency": int(save_every) if save_every else max(1, epochs * n_samples)},
        "lora_layers": {"targets": lora},
    }
    if 0 < ram_gb < 24:
        cfg["cache_limit_gb"] = 1.0 if ram_gb < 16 else 2.0
    return cfg


def add_metadata(path, extra):
    """Add entries to a .safetensors file's metadata without touching the tensors (rewrites only the header)."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = dict(header.get("__metadata__") or {})
    meta.update({k: str(v) for k, v in extra.items()})
    header["__metadata__"] = meta
    new = json.dumps(header, separators=(",", ":")).encode()
    new += b" " * ((8 - len(new) % 8) % 8)  # keep the data aligned like the safetensors writer does
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(struct.pack("<Q", len(new)) + new + raw[8 + n:])
    tmp.replace(path)


def latest_adapter(out_dir):
    """The newest *_adapter.safetensors inside mflux's checkpoint zips (it saves to a dated sub-folder)."""
    zips = sorted(Path(out_dir).rglob("*_checkpoint.zip"), key=lambda p: (p.stat().st_mtime, p.name))
    for z in reversed(zips):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith("_adapter.safetensors")]
            if names:
                return z, sorted(names)[-1]
    return None, None


def parse_progress(line):
    """(done, total, seconds_per_step or None) from a tqdm line, else None."""
    m = None
    for m in _PROGRESS.finditer(line):
        pass
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    sps = float(m.group(3)) if m.group(3) else (1.0 / float(m.group(4)) if m.group(4) and float(m.group(4)) > 0 else None)
    return done, total, sps


def train(*, spec, items, caption_for, args, name, final_path, work_dir, emit, cancel_event, ram_gb, pause_event=None):
    import mlx_engine
    from PIL import Image, ImageOps
    if not mlx_engine.installed():
        raise TrainError("The fast engine isn't installed yet: make one picture with the ERNIE fast-engine model first.")
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(spec["mlx_repo"], local_files_only=True)
    except Exception:
        raise TrainError(f"{spec['label']} isn't downloaded yet: download it in the Models tab (or make one picture with it) first.")

    trigger = (args.get("trigger") or "").strip()
    res = int(args.get("resolution") or (512 if 0 < ram_gb < 16 else 768))
    res = max(256, min(1024, res - res % 16))
    steps = int(args.get("steps") or 1000)
    lr = float(args.get("lr") or 1e-4)
    rank = int(args.get("rank") or 16)
    flip = bool(args.get("flip", True))

    # 1. data folder: every picture with its caption (trigger word first), plus mirrored copies
    shutil.rmtree(work_dir, ignore_errors=True)
    data_dir = Path(work_dir) / "data"
    out_dir = Path(work_dir) / "out"
    data_dir.mkdir(parents=True)
    n = 0
    for i, it in enumerate(items):
        if cancel_event.is_set():
            raise Cancelled()
        emit({"type": "status", "msg": f"Preparing images {i + 1}/{len(items)}…"})
        try:
            img = ImageOps.exif_transpose(Image.open(__import__("pictures").prepare(it["path"]))).convert("RGB")
        except Exception:
            continue
        if it.get("fit") and it["fit"] != "center":  # framed by hand in the Train tab: train on exactly that square
            import trainer
            img = trainer.square(img, res, it["fit"])
        cap = caption_for(it, trigger)
        for k, v in enumerate([img, ImageOps.mirror(img)] if flip else [img]):
            stem = f"{i:04d}_{k}"
            v.save(data_dir / f"{stem}.png")
            (data_dir / f"{stem}.txt").write_text(cap, encoding="utf-8")
            n += 1
    if n < 3:
        raise TrainError("None of the images could be opened.")

    cfg = build_config(model_path=model_path, data_dir=data_dir, out_dir=out_dir, steps=steps, n_samples=n, lr=lr,
                       rank=rank, res=res, targets=args.get("targets"), save_every=int(args.get("save_every") or 0),
                       ram_gb=ram_gb)
    cfg_path = Path(work_dir) / "train.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # 2. run mflux's trainer in the fast engine's Python
    emit({"type": "status", "msg": "Loading ERNIE for training (fast engine)…"})
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    proc = subprocess.Popen([str(mlx_engine.PY), "-m", "mflux.models.common.cli.train", "--config", str(cfg_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(work_dir), env=env)
    total_iters = cfg["training_loop"]["num_epochs"] * n
    started, last_emit, tail, buf = time.time(), 0.0, [], b""
    cancelled = False
    if pause_event is not None:
        def hold():  # Pause: suspend the trainer process (it keeps its memory); Resume / Stop wake it
            frozen = False
            while proc.poll() is None:
                want = pause_event.is_set() and not cancel_event.is_set()
                if want != frozen:
                    try:
                        os.kill(proc.pid, signal.SIGSTOP if want else signal.SIGCONT)
                        frozen = want
                    except OSError:
                        return
                time.sleep(0.25)
        threading.Thread(target=hold, daemon=True).start()
    try:
        while True:
            if cancel_event.is_set() and not cancelled:
                cancelled = True
                proc.terminate()
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                pr = parse_progress(line)
                if pr:
                    done, total, sps = pr
                    total_iters = total or total_iters
                    if time.time() - last_emit > 1.0:
                        emit({"type": "progress", "image": 1, "images": 1, "step": done, "steps": total_iters})
                        if sps:
                            emit({"type": "train_status", "sec_per_step": round(sps, 2), "step": done})
                        last_emit = time.time()
                    continue
                tail = (tail + [line])[-25:]
                print("[mlx-train]", line, file=sys.stderr, flush=True)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()

    # 3. the LoRA: the last saved checkpoint (also after Stop, if one was saved)
    zpath, member = latest_adapter(out_dir)
    if cancelled and not member:
        raise Cancelled()
    if not member:
        msg = "\n".join(tail[-8:])
        if code and code < 0 or "memory" in msg.lower() or "Metal" in msg:
            msg = "The Mac ran out of memory. Try 512 px, rank 8 and “attention layers”, and close other apps.\n\n" + msg
        raise TrainError("ERNIE training on the fast engine failed:\n" + msg)
    with zipfile.ZipFile(zpath) as zf, zf.open(member) as src, open(final_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    step_done = int(re.match(r"(\d+)", Path(member).name).group(1)) if re.match(r"(\d+)", Path(member).name) else steps
    add_metadata(final_path, {
        "beestudio_family": "ernie", "protoart_engine": "mlx", "ss_base_model_version": "ernie_image",
        "modelspec.architecture": "ernie-image/lora", "modelspec.trigger_phrase": trigger,
        "ss_output_name": name, "ss_steps": step_done, "ss_resolution": res, "ss_network_dim": rank,
        "ss_learning_rate": lr, "ss_sd_model_name": spec["label"],
    })
    shutil.rmtree(work_dir, ignore_errors=True)
    return {"name": name, "path": str(final_path), "steps": step_done, "seconds": round(time.time() - started),
            "format": "mflux (fast engine)", "stopped_early": cancelled}
