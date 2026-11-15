"""Plan and execute Qwen Image 2.1 generation with explicit model policies."""

import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..api import GenerationResult, QwenOptions
from . import assets
from .lora import inspect_adapter
from .runtime import locate, run_worker

TUNING = {
    "resident": ("resident-cache", [4096, 40, 32, 256, 0, 0, 32, 0, 0, 1, 0]),
    "prefetch": ("prefetch", [256, 8, 4, 64, 0, 0, 8, 0, 0, 0, 0]),
    "micro-fast": ("micro", [4096, 40, 32, 256, 1, 0, 32, 1, 0, 1, 0]),
    "micro": ("micro", [256, 8, 4, 64, 0, 0, 8, 0, 0, 0, 0]),
}


def finite(value, name, low, high):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def validate(prompt, width, height, seed, references, o):
    if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 24000:
        raise ValueError("Provide a nonempty prompt of at most 24000 characters")
    if any(
        type(v) is not int or not 256 <= v <= 4096 or v % 32 for v in (width, height)
    ):
        raise ValueError("Dimensions must be multiples of 32 from 256 to 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if isinstance(references, (str, Path)) or len(references) > 10:
        raise ValueError("Qwen accepts an ordered list of up to 10 references")
    if not isinstance(o, QwenOptions):
        raise ValueError("qwen must be QwenOptions")
    if o.memory_mode not in TUNING:
        raise ValueError(f"Unknown memory mode: {o.memory_mode}")
    if o.weight_cache not in ("system", "uncached"):
        raise ValueError("weight_cache must be system or uncached")
    if o.prepared and not o.prepare_weights:
        raise ValueError("prepared and prepare_weights=False conflict")
    if o.profile not in assets.PROFILES:
        raise ValueError(f"Unknown profile: {o.profile}")
    if o.mode not in ("edit", "variation"):
        raise ValueError("Reference mode must be edit or variation")
    if o.mode == "variation" and len(references) != 1:
        raise ValueError("Variation needs exactly one input image")
    if type(o.steps) is not int or not 1 <= o.steps <= 100:
        raise ValueError("Steps must be in [1,100]")
    if o.profile == "complete" and o.steps != 8 and o.sample_sigmas is None:
        raise ValueError(
            "The Turbo checkpoint has eight fixed steps; use turbo3/turbo4 or custom sigmas"
        )
    finite(o.guidance_scale, "guidance_scale", 0, 100)
    finite(o.profile_strength, "profile_strength", -20, 20)
    finite(o.timeout, "timeout", 0.01, 86400)
    finite(o.strength, "strength", 0.00001, 1)
    if not isinstance(o.negative_prompt, str):
        raise ValueError("Negative prompt must be a string")
    if (
        type(o.reference_resolution) is not int
        or not 256 <= o.reference_resolution <= 1024
    ):
        raise ValueError("reference_resolution must be in [256,1024]")
    if o.auto_strength and o.profile != "lightning":
        raise ValueError("Automatic LoRA strength applies only to Lightning")
    if len(o.loras) + (o.profile in ("lightning", "dmd2", "turbo8")) > 8:
        raise ValueError("At most eight LoRAs, including the sampling preset")
    for lora in o.loras:
        finite(lora.scale, "LoRA scale", -20, 20)
        inspect_adapter(Path(lora.path).expanduser())
    if o.sample_sigmas is not None:
        if o.profile != "complete":
            raise ValueError("Custom sigmas cannot override a sampling preset")
        if len(o.sample_sigmas) != o.steps:
            raise ValueError("Provide one sigma per denoising step")
        for sigma in o.sample_sigmas:
            finite(sigma, "sigma", 0.000001, 1)
        if any(a <= b for a, b in zip(o.sample_sigmas, o.sample_sigmas[1:])):
            raise ValueError("Sigmas must be strictly descending")
    if any(
        int(n) < 1 or int(n) > len(references)
        for n in re.findall(r"<image(\d+)>", prompt, re.I)
    ):
        raise ValueError("Prompt refers to a missing reference image")


def encode_references(references):
    encoded = []
    total = 0
    for value in references:
        with Image.open(Path(value).expanduser()) as image:
            image.load()
            if image.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("References must be PNG, JPEG or WebP")
            out = io.BytesIO()
            image.save(out, format="PNG")
            raw = out.getvalue()
            total += len(raw)
            if len(raw) > 20 * 1024**2 or total > 100 * 1024**2:
                raise ValueError("Reference images exceed the request size budget")
            encoded.append(base64.b64encode(raw).decode())
    return encoded


def pixels(path, transparent=False):
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        spec = json.loads(f.read(n))["value"]
        begin, end = spec["data_offsets"]
        f.seek(8 + n + begin)
        raw = f.read(end - begin)
    dtype = {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}[spec["dtype"]]
    values = np.frombuffer(raw, dtype=dtype)
    if spec["dtype"] == "BF16":
        values = (values.astype("<u4") << 16).view("<f4")
    values = values.reshape(spec["shape"]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Image contains non-finite pixels")
    array = np.uint8(np.clip(values[0].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255)
    return Image.fromarray(array if transparent else array[:, :, :3])


def generate(
    prompt,
    *,
    models,
    width,
    height,
    seed,
    references,
    backend,
    options,
    on_progress=None,
    cancel=None,
):
    o = options
    validate(prompt, width, height, seed, references, o)
    model = assets.check(models)
    binary = locate(o.runtime)
    if backend not in ("auto", "portable", "w8a8"):
        raise ValueError("Unknown compute backend")
    chip = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    ).strip()
    from .._stage import select_backend

    selected = select_backend(backend, chip)
    if o.weight_cache == "uncached" and (
        selected != "w8a8"
        or o.memory_mode not in ("micro", "micro-fast")
        or references
        or o.loras
        or o.private
        or o.transparent
        or o.profile != "complete"
        or o.sample_sigmas is not None
        or o.negative_prompt
        or o.guidance_scale != 1
    ):
        raise ValueError(
            "Uncached reads currently require staged W8A8 Turbo text-to-image in micro or micro-fast mode"
        )
    native_backend = "w8a8" if selected == "w8a8" else "packed"
    cache = (
        Path(o.cache or Path.home() / ".cache/h3-image-mlx/qwen").expanduser().resolve()
    )
    cache.mkdir(parents=True, exist_ok=True)
    if references:
        model = assets.ensure_vision(model, cache, cancel)
    model, profile, preset_paths, preset_scales = assets.resolve_profile(
        model, o, cache, width, height
    )
    body = dict(
        prompt=prompt,
        size=f"{width}x{height}",
        seed=seed,
        steps=o.steps,
        guidance_scale=o.guidance_scale,
        negative_prompt=o.negative_prompt,
        transparent=o.transparent,
        ref_resolution=o.reference_resolution,
    )
    body.update(profile)
    if o.sample_sigmas is not None:
        body["sample_sigmas"] = list(o.sample_sigmas)
    refs = encode_references(references)
    if refs:
        body.update(image=refs[0], mode=o.mode, strength=o.strength)
        if len(refs) > 1:
            body["ref_images"] = refs[1:]
    paths = preset_paths + [str(Path(x.path).expanduser().resolve()) for x in o.loras]
    if paths:
        body.update(
            lora_paths=paths, lora_scales=preset_scales + [x.scale for x in o.loras]
        )
    mode, knobs = TUNING[o.memory_mode]
    if native_backend == "packed" and mode == "resident-cache":
        mode = "resident"
    prepared = None
    preparation = []

    def command(stage, source, output, blocks=None):
        return [
            binary,
            stage,
            model,
            source,
            output,
            body["size"],
            mode,
            native_backend,
            blocks or "-",
            seed,
            32,
            *knobs,
        ]

    def execute(stage, source, output, blocks=None, body_bytes=None):
        if on_progress:
            on_progress(dict(stage=stage, step=0, total=body["steps"]))
        env = dict(os.environ)
        env.pop("SHENGTU_LIGHTNING_AUTO_ADAPTER", None)
        if o.auto_strength:
            env["SHENGTU_LIGHTNING_AUTO_ADAPTER"] = paths[0]
        if o.weight_cache == "uncached":
            library = binary.parent / "lib/aligned_nocache.dylib"
            if not library.is_file():
                raise FileNotFoundError(
                    "This runtime lacks aligned uncached I/O support; rebuild the native package"
                )
            roots = [
                model.resolve(),
                *(p.resolve() for p in model.iterdir() if p.is_dir()),
            ]
            if blocks:
                roots.append(Path(blocks).resolve())
            env.update(
                DYLD_INSERT_LIBRARIES=str(library),
                IMAGE_MLX_UNCACHED_ROOTS="\n".join(map(str, roots)),
            )
        out, metric = run_worker(
            command(stage, source, output, blocks),
            input_bytes=body_bytes,
            timeout=o.timeout,
            cancel=cancel,
            on_progress=on_progress,
            env=env,
        )
        metric["stage"] = stage
        return out, metric

    if selected == "w8a8" and o.prepare_weights:
        prepared = (
            Path(o.prepared).expanduser().resolve()
            if o.prepared
            else cache / "prepared" / assets.identity(model / "transformer")
        )
        with assets.locked(prepared.parent / (prepared.name + ".lock")):
            ready = all(
                (prepared / f"block-{i:02d}.safetensors").is_file() for i in range(32)
            )
            if ready:
                assets.validate_prepared(prepared, model)
            if not ready:
                if o.prepared:
                    raise ValueError(
                        "Explicit prepared cache must contain all 32 blocks"
                    )
                if shutil.disk_usage(cache).free < 8_000_000_000:
                    raise OSError("Preparing Qwen W8A8 blocks requires 8 GB free disk")
                with tempfile.TemporaryDirectory(
                    dir=prepared.parent, prefix="prepare-"
                ) as tmp:
                    _, metric = execute("prepare", "-", tmp)
                    preparation.append(metric)
                    if not all(
                        (Path(tmp) / f"block-{i:02d}.safetensors").is_file()
                        for i in range(32)
                    ):
                        raise RuntimeError("Incomplete prepared blocks")
                    (Path(tmp) / "manifest.json").write_text(
                        json.dumps(
                            dict(
                                format=1,
                                transformer_identity=assets.identity(
                                    model / "transformer"
                                ),
                            )
                        )
                    )
                    prepared.mkdir(exist_ok=True)
                    for file in Path(tmp).iterdir():
                        file.replace(prepared / file.name)
    staged = not (
        refs
        or paths
        or o.private
        or body["negative_prompt"]
        or body["guidance_scale"] != 1
        or body["steps"] != 8
        or body.get("sample_sigmas")
        or o.profile != "complete"
        or o.transparent
    )
    started = time.monotonic()
    stages = []
    if staged:
        with tempfile.TemporaryDirectory(prefix="qwen-stages-") as tmp:
            work = Path(tmp)
            (work / "prompt.txt").write_text(prompt)
            for stage, source in (
                ("text", "prompt.txt"),
                ("dit", "text.safetensors"),
                ("vae", "dit.safetensors"),
            ):
                _, metric = execute(
                    stage,
                    work / source,
                    work / (stage + ".safetensors"),
                    prepared if stage == "dit" else None,
                )
                stages.append(metric)
            image = pixels(work / "vae.safetensors")
    else:
        out, metric = execute("request", "-", "-", prepared, json.dumps(body).encode())
        stages.append(metric)
        with Image.open(io.BytesIO(out)) as opened:
            image = opened.convert("RGBA" if o.transparent else "RGB")
    return GenerationResult(
        image,
        dict(
            model="qwen-image-2.1-turbo",
            backend=selected,
            memory_mode=o.memory_mode,
            profile=o.profile,
            steps=body["steps"],
            seed=seed,
            stages=stages,
            preparation=preparation,
            worker_peak_bytes=max(x.get("peak_bytes", 0) for x in stages),
            disk_read_bytes=sum(x.get("disk_read_bytes", 0) for x in stages),
            generation_seconds=time.monotonic() - started,
            execution="staged" if staged else "request",
            weight_read_policy=o.weight_cache,
        ),
    )
