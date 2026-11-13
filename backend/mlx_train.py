"""LoRA training on the fast engine (MLX, via mflux's trainer): ERNIE, FLUX.2 Klein (4B / 9B) and Z-Image Turbo.

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


def _targets(rows, rank):
    out = []
    for path, end in rows:
        t = {"module_path": path, "rank": int(rank)}
        if end:
            t["blocks"] = {"start": 0, "end": end}
        out.append(t)
    return out


def _flux2_targets(double, single, full):
    rows = [(f"transformer_blocks.{{block}}.attn.{n}", double) for n in ("to_q", "to_k", "to_v", "to_out")]
    rows += [(f"single_transformer_blocks.{{block}}.attn.{n}", single) for n in ("to_qkv_mlp_proj", "to_out")]
    if full:
        rows += [(f"transformer_blocks.{{block}}.attn.{n}", double) for n in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")]
        rows += [(f"transformer_blocks.{{block}}.{n}", double) for n in ("ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out")]
    return rows


def _zimage_targets(full):
    rows = [(f"layers.{{block}}.attention.{n}", 30) for n in ("to_q", "to_k", "to_v", "to_out.0")]
    if full:
        rows += [(f"layers.{{block}}.feed_forward.{n}", 30) for n in ("w1", "w2", "w3")]
        rows += [("cap_embedder.1", None), ("all_final_layer.2-1.linear", None)]
    return rows


# what each fast-engine model trains with (mflux's own example settings, LoRA targets from its examples / READMEs)
KINDS = {
    "ernie": {"name": "ERNIE", "model": lambda s: "ernie-image-turbo", "steps": 8, "guidance": 1.0, "t": (1, 8),
              "targets": lambda full: [(f"layers.{{block}}.{n}", ERNIE_BLOCKS) for n in (ERNIE_TARGETS_FULL if full else ERNIE_TARGETS_ATTENTION)]
              + ([("text_proj", None)] if full else []),
              "meta": {"ss_base_model_version": "ernie_image", "modelspec.architecture": "ernie-image/lora"}},
    "flux2": {"name": "FLUX.2 Klein", "model": lambda s: s.get("mlx_base") or "flux2-klein-4b", "steps": 4, "guidance": 1.0,
              "t": (1, 4), "targets": None,  # set per size below
              "meta": {"ss_base_model_version": "flux2_klein", "modelspec.architecture": "flux-2-klein/lora"}},
    "zimage": {"name": "Z-Image Turbo", "model": lambda s: "z-image-turbo", "steps": 9, "guidance": 0.0, "t": (4, 9),
               "targets": _zimage_targets,
               "meta": {"ss_base_model_version": "z_image_turbo", "modelspec.architecture": "z-image/lora"}},
}
# Z-Image Turbo needs this helper LoRA while training (mflux loads it itself; downloaded beforehand, in the app)
ZIMAGE_ADAPTER = ("ostris/zimage_turbo_training_adapter", "zimage_turbo_training_adapter_v2.safetensors")


def kind_of(spec):
    return spec.get("family") if spec.get("engine") == "mlx" and spec.get("family") in KINDS else None


def lora_rows(spec, targets):
    full = targets == "full"
    fam = spec["family"]
    if fam == "flux2":
        big = "9b" in str(spec.get("mlx_base") or "").lower()
        return _flux2_targets(8 if big else 5, 24 if big else 20, full)
    return KINDS[fam]["targets"](full)

_PROGRESS = re.compile(r"(\d+)/(\d+) \[[^\]]*?(?:([\d.]+)s/it|([\d.]+)it/s)")


class TrainError(Exception):
    pass


class Cancelled(Exception):
    pass


def build_config(*, model_path, data_dir, out_dir, steps, n_samples, lr, rank, res, targets, save_every, ram_gb, seed=42,
                 spec=None):
    """mflux training config (same layout as its train_*.json examples)."""
    spec = spec or {"family": "ernie", "engine": "mlx"}
    k = KINDS[spec["family"]]
    epochs = max(1, math.ceil(steps / max(1, n_samples)))
    lora = _targets(lora_rows(spec, targets), rank)
    small = 0 < ram_gb < 32
    cfg = {
        "model": k["model"](spec),
        "model_path": str(model_path),
        "data": str(data_dir),
        "seed": int(seed),
        "steps": k["steps"], "guidance": k["guidance"],   # the model's own sampling settings (previews)
        "quantize": None,               # the fast-engine model is already 4-bit
        "max_resolution": int(res),
        "low_ram": small,               # encoded data kept on disk instead of in memory
        "gradient_checkpointing": 0 < ram_gb < 64,  # ~30% slower, much less memory
        "training_loop": {"num_epochs": epochs, "batch_size": 1, "timestep_low": k["t"][0], "timestep_high": k["t"][1]},
        "optimizer": {"name": "AdamW", "learning_rate": float(lr)},
        # saved every 50 steps (or as set), so ■ Stop keeps the progress and ▶ Continue can carry on from it
        "checkpoint": {"output_path": str(out_dir), "save_frequency": int(save_every) if save_every else min(50, max(1, epochs * n_samples))},
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
        try:  # one being written when Stop was pressed is incomplete: use the one before it
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith("_adapter.safetensors")]
                if names and zf.testzip() is None:
                    return z, sorted(names)[-1]
        except (zipfile.BadZipFile, OSError, EOFError):
            continue
    return None, None


def prune_checkpoints(out_dir, keep=2):
    """Keep only the newest few checkpoint zips (each holds the LoRA and the optimizer: tens of MB to ~1 GB)."""
    zips = sorted(Path(out_dir).rglob("*_checkpoint.zip"), key=lambda p: (p.stat().st_mtime, p.name))
    for z in zips[:-keep]:
        try:
            z.unlink()
        except OSError:
            pass


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
        raise TrainError(f"The fast engine isn't installed yet: make one picture with {spec['label']} first.")
    kind = KINDS[spec["family"]]
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

    resume_zip = args.get("mlx_resume")
    if resume_zip and not os.path.exists(resume_zip):
        raise TrainError("The stopped training's saved progress isn't there any more: start it again.")
    data_dir = Path(work_dir) / "data"
    out_dir = Path(work_dir) / "out"
    n = 0
    if not resume_zip:
        # 1. data folder: every picture with its caption (trigger word first), plus mirrored copies
        shutil.rmtree(work_dir, ignore_errors=True)
        data_dir.mkdir(parents=True)
    for i, it in enumerate([] if resume_zip else items):
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
    if resume_zip:
        n = len(list(data_dir.glob("*.png")))
    if n < 3:
        raise TrainError("None of the images could be opened.")

    cfg = build_config(model_path=model_path, data_dir=data_dir, out_dir=out_dir, steps=steps, n_samples=n, lr=lr,
                       rank=rank, res=res, targets=args.get("targets"), save_every=int(args.get("save_every") or 0),
                       ram_gb=ram_gb, spec=spec)
    if spec["family"] == "zimage" and not resume_zip:
        # its training helper LoRA: fetched here (this app's downloads get past a VPN that blocks the trainer's Python)
        import net
        from huggingface_hub import hf_hub_download
        emit({"type": "status", "msg": "Downloading Z-Image's training helper (once, small)…"})
        net.set_online(True)
        try:
            hf_hub_download(*ZIMAGE_ADAPTER)
        except Exception as e:
            raise TrainError(f"Couldn't download Z-Image's training helper ({e}). Check the internet connection.")
        finally:
            net.set_online(False)
    cfg_path = Path(work_dir) / "train.json"
    if not resume_zip:
        cfg_path.write_text(json.dumps(cfg, indent=2))

    # 2. run mflux's trainer in the fast engine's Python
    emit({"type": "status", "msg": f"Loading {kind['name']} to carry on training (fast engine)…" if resume_zip
          else f"Loading {kind['name']} for training (fast engine)…"})
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    how = ["--resume", str(resume_zip)] if resume_zip else ["--config", str(cfg_path)]
    proc = subprocess.Popen([str(mlx_engine.PY), "-m", "mflux.models.common.cli.train", *how],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(work_dir), env=env)
    total_iters = cfg["training_loop"]["num_epochs"] * n
    started, last_emit, tail, buf = time.time(), 0.0, [], b""
    cancelled = False
    last_prune = time.time()
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
                if time.time() - last_prune > 30:
                    prune_checkpoints(out_dir)
                    last_prune = time.time()
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
        raise TrainError(f"{kind['name']} training on the fast engine failed:\n" + msg)
    with zipfile.ZipFile(zpath) as zf, zf.open(member) as src, open(final_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    step_done = int(re.match(r"(\d+)", Path(member).name).group(1)) if re.match(r"(\d+)", Path(member).name) else steps
    add_metadata(final_path, {
        "beestudio_family": spec["family"], "protoart_engine": "mlx", **kind["meta"], "modelspec.trigger_phrase": trigger,
        "ss_output_name": name, "ss_steps": step_done, "ss_resolution": res, "ss_network_dim": rank,
        "ss_learning_rate": lr, "ss_sd_model_name": spec["label"],
    })
    if cancelled:  # kept for ▶ Continue: the pictures, the settings and the last good checkpoint (with the optimizer)
        for z in Path(out_dir).rglob("*_checkpoint.zip"):
            if z != zpath:
                try:
                    z.unlink()
                except OSError:
                    pass
    else:
        shutil.rmtree(work_dir, ignore_errors=True)
    return {"name": name, "path": str(final_path), "steps": step_done, "seconds": round(time.time() - started),
            "format": "mflux (fast engine)", "stopped_early": cancelled, "total_steps": total_iters,
            **({"checkpoint": str(zpath)} if cancelled else {})}
