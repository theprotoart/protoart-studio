"""Qwen model assets and immutable transformer views."""

import fcntl
import hashlib
import json
import math
import shutil
from contextlib import contextmanager
from pathlib import Path

from ..download import fetch_manifest
from ..models import sha256

MANIFEST = json.loads(Path(__file__).with_name("weights.json").read_text())
ADAPTERS = json.loads(Path(__file__).with_name("adapters.json").read_text())
PROFILES = ("complete", "turbo4", "turbo3", "standard", "lightning", "dmd2", "turbo8")


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def check(directory, manifest=MANIFEST, hashes=False):
    root = Path(directory).expanduser().resolve()
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing Qwen asset: {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Unexpected Qwen asset size: {path}")
        if hashes and sha256(path) != item["sha256"]:
            raise ValueError(f"Qwen asset checksum mismatch: {path}")
    return root


def fetch(directory, source="modelscope"):
    return fetch_manifest(directory, MANIFEST, source)


def fetch_profile(directory, profile, source="modelscope"):
    if profile not in PROFILES:
        raise ValueError(f"Unknown Qwen profile: {profile}")
    if profile in ("complete", "turbo4", "turbo3"):
        return Path(directory)
    root = Path(directory)
    for name in ("base", *(() if profile == "standard" else (profile,))):
        manifest = ADAPTERS[name]
        # Some third-party LoRAs publish only on Hugging Face.
        selected = (
            source if all(source in e for e in manifest["files"]) else "huggingface"
        )
        fetch_manifest(root / name, manifest, selected)
    return root


def identity(directory):
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise ValueError(f"Missing transformer weights: {directory}")
    state = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    config = Path(directory) / "config.json"
    if config.is_file():
        state.append((str(config.resolve()), sha256(config)))
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


def validate_prepared(directory, model):
    """An explicit prepared pack must identify the same unchanged transformer."""
    manifest = Path(directory) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            "Prepared weights need manifest.json identifying their transformer"
        )
    data = json.loads(manifest.read_text())
    if "transformer_identity" in data:
        if data["transformer_identity"] != identity(model / "transformer"):
            raise ValueError("Prepared weights belong to another transformer")
    else:
        source = Path(data["model"])
        if (source / "transformer").resolve() != (model / "transformer").resolve():
            raise ValueError("Prepared weights belong to another transformer")
        for name, size, mtime in data["files"]:
            if name.startswith("transformer/"):
                stat = (model / name).stat()
                if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
                    raise ValueError("Prepared weights are stale")


def resolve_profile(model, options, cache, width, height):
    """Return model root, effective schedule/options, and ordered preset adapters."""
    profile = options.profile
    body = {}
    paths = []
    scales = []
    schedules = {
        "turbo4": [1.0, 0.89508, 0.704534, 0.414568],
        "turbo3": [1.0, 0.704534, 0.414568],
    }
    if profile in schedules:
        body.update(sample_sigmas=schedules[profile], steps=len(schedules[profile]))
    if profile in ("complete", "turbo4", "turbo3"):
        return model, body, paths, scales
    if not options.profile_assets:
        raise ValueError(f"{profile} needs profile_assets; fetch the preset first")
    assets = Path(options.profile_assets).expanduser().resolve()
    base = check(assets / "base", ADAPTERS["base"])
    index = json.loads((model / "model_index.json").read_text())
    index.pop("sample_sigmas", None)
    if profile != "standard":
        preset = check(assets / profile, ADAPTERS[profile])
        adapter = preset / "adapter.safetensors"
        if profile == "lightning":
            e = math.exp(
                0.5 + ((width // 16) * (height // 16) - 256) * 0.4 / (8192 - 256)
            )
            body.update(
                steps=4,
                sample_sigmas=[e / (e + 1 / s - 1) for s in (1, 0.75, 0.5, 0.25)],
            )
        elif profile == "dmd2":
            body.update(steps=2, sample_sigmas=[1.0, 0.5])
        elif profile == "turbo8":
            from .turbo_adapter import prepare_turbo

            keyed = (
                cache
                / "turbo"
                / hashlib.sha256(
                    json.dumps(
                        [
                            identity(base),
                            str(adapter),
                            adapter.stat().st_mtime_ns,
                            options.profile_strength,
                        ]
                    ).encode()
                ).hexdigest()
            )
            with locked(keyed / ".lock"):
                base, adapter = prepare_turbo(
                    keyed, base, adapter, options.profile_strength
                )
            # Non-block Turbo corrections and block adapter use the same strength.
            body.update(
                steps=8,
                sample_sigmas=json.loads((model / "model_index.json").read_text())[
                    "sample_sigmas"
                ],
            )
        if profile == "lightning" and options.auto_strength:
            auto = cache / "auto" / sha256(adapter) / "lightning-auto.safetensors"
            with locked(auto.parent / ".lock"):
                if not auto.exists():
                    auto.symlink_to(adapter)
            adapter = auto
        paths.append(str(adapter))
        scales.append(1.0 if options.auto_strength else options.profile_strength)
        body.update(guidance_scale=1.0, negative_prompt="")
    key = hashlib.sha256(
        json.dumps([str(model), identity(base), index]).encode()
    ).hexdigest()
    view = cache / "views" / key
    with locked(view.parent / (key + ".lock")):
        if not (view / "model_index.json").is_file():
            view.mkdir(parents=True, exist_ok=True)
            for child in model.iterdir():
                if child.name in ("transformer", "model_index.json"):
                    continue
                target = view / child.name
                if not target.exists():
                    target.symlink_to(
                        child.resolve(), target_is_directory=child.is_dir()
                    )
            if not (view / "transformer").exists():
                (view / "transformer").symlink_to(base, target_is_directory=True)
            (view / "model_index.json").write_text(json.dumps(index))
    return view, body, paths, scales


def ensure_vision(model, cache, cancel=None):
    if (model / "vision/model.safetensors").is_file():
        return model
    from .vision_assets import extract_vision

    src = model / "text_encoder/model.safetensors"
    key = hashlib.sha256(
        f"{src.resolve()}:{src.stat().st_mtime_ns}:{src.stat().st_size}".encode()
    ).hexdigest()
    view = cache / "vision" / key
    with locked(view.parent / (key + ".lock")):
        if not (view / "vision/model.safetensors").is_file():
            if shutil.disk_usage(cache).free < 800_000_000:
                raise OSError("Vision extraction needs 800 MB free disk space")
            extract_vision(src, view / "vision/model.safetensors", cancel)
        for child in model.iterdir():
            if child.name == "vision":
                continue
            dest = view / child.name
            if not dest.exists():
                dest.symlink_to(child.resolve(), target_is_directory=child.is_dir())
    return view
