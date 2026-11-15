"""CLI model selection shares the same typed requests as the Python API."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .api import MODEL_IDS, Lora, QwenOptions, generate_result
from .models import check_models


def main(argv=None):
    parser = argparse.ArgumentParser(description="H3 and Qwen Image on Apple Silicon")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate or edit ordered references")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--reference", type=Path, action="append", default=[])
    gen.add_argument("--backend", choices=("auto", "w8a8", "portable"), default="auto")
    gen.add_argument(
        "--dit-mode", choices=("auto", "stream", "resident"), default="auto"
    )
    gen.add_argument("--metrics", type=Path)
    gen.add_argument("--progress", action="store_true")
    q = gen.add_argument_group("Qwen Image 2.1 options")
    q.add_argument(
        "--memory-mode", choices=("resident", "prefetch", "micro-fast", "micro")
    )
    q.add_argument("--runtime", type=Path)
    q.add_argument("--cache", type=Path)
    q.add_argument("--prepared", type=Path)
    q.add_argument(
        "--no-prepare",
        action="store_true",
        help="Read original weights without an extra prepared DiT copy",
    )
    q.add_argument("--weight-cache", choices=("system", "uncached"))
    q.add_argument("--steps", type=int)
    q.add_argument("--cfg", type=float)
    q.add_argument("--negative-prompt")
    q.add_argument(
        "--profile",
        choices=(
            "complete",
            "turbo4",
            "turbo3",
            "standard",
            "lightning",
            "dmd2",
            "turbo8",
        ),
    )
    q.add_argument("--profile-assets", type=Path)
    q.add_argument("--profile-strength", type=float)
    q.add_argument("--auto-strength", action="store_true")
    q.add_argument("--lora", type=Path, action="append", default=[])
    q.add_argument("--lora-scale", type=float, action="append", default=[])
    q.add_argument("--sigmas", type=float, nargs="+")
    q.add_argument("--reference-mode", choices=("edit", "variation"))
    q.add_argument("--strength", type=float)
    q.add_argument("--reference-resolution", type=int)
    q.add_argument("--transparent", action="store_true")
    q.add_argument(
        "--private",
        action="store_true",
        help="Pass request data through pipes, without temporary prompt/image files",
    )
    get = commands.add_parser("fetch", help="Download pinned model weights")
    get.add_argument(
        "--source", choices=("modelscope", "huggingface"), default="modelscope"
    )
    verify = commands.add_parser("verify", help="Verify pinned file hashes")
    for command in (gen, get, verify):
        command.add_argument("--models", type=Path, required=True)
        command.add_argument("--model", choices=MODEL_IDS, default="h3")
        command.add_argument("--precision", choices=("int8", "bf16"), default="int8")
    presets = commands.add_parser(
        "fetch-profile", help="Download a Qwen sampling preset and base DiT"
    )
    presets.add_argument(
        "--profile", required=True, choices=("standard", "lightning", "dmd2", "turbo8")
    )
    presets.add_argument("--directory", type=Path, required=True)
    presets.add_argument(
        "--source", choices=("modelscope", "huggingface"), default="modelscope"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-profile":
            from .qwen.assets import fetch_profile

            print(fetch_profile(args.directory, args.profile, args.source))
            return 0
        if args.model != "h3" and args.precision != "int8":
            raise ValueError("The Qwen model pack uses Q8 weights")
        if args.command == "fetch":
            if args.model == "h3":
                from .download import fetch

                print(fetch(args.models, args.source, precision=args.precision))
            else:
                from .qwen.assets import fetch

                print(fetch(args.models, args.source))
        elif args.command == "verify":
            if args.model == "h3":
                print(check_models(args.models, hashes=True, precision=args.precision))
            else:
                from .qwen.assets import check

                print(check(args.models, hashes=True))
        else:
            if args.output.suffix.lower() != ".png":
                raise ValueError("Output must be a .png file")
            fields = {
                "memory_mode": args.memory_mode,
                "runtime": args.runtime,
                "cache": args.cache,
                "prepared": args.prepared,
                "steps": args.steps,
                "guidance_scale": args.cfg,
                "negative_prompt": args.negative_prompt,
                "profile": args.profile,
                "profile_assets": args.profile_assets,
                "profile_strength": args.profile_strength,
                "sample_sigmas": args.sigmas,
                "mode": args.reference_mode,
                "strength": args.strength,
                "reference_resolution": args.reference_resolution,
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            if args.no_prepare:
                fields["prepare_weights"] = False
            if args.weight_cache:
                fields["weight_cache"] = args.weight_cache
            for name in ("auto_strength", "transparent", "private"):
                if getattr(args, name):
                    fields[name] = True
            if args.lora_scale and len(args.lora_scale) != len(args.lora):
                raise ValueError(
                    "Provide one --lora-scale per --lora, or omit all scales for 1.0"
                )
            if args.lora:
                fields["loras"] = tuple(
                    Lora(p, s)
                    for p, s in zip(
                        args.lora, args.lora_scale or [1.0] * len(args.lora)
                    )
                )
            if args.model == "h3" and fields:
                raise ValueError("Qwen options require --model qwen-image-2.1-turbo")
            progress = (
                (lambda e: print(json.dumps(e), file=sys.stderr, flush=True))
                if args.progress
                else None
            )
            result = generate_result(
                args.prompt,
                models=args.models,
                model=args.model,
                width=args.width,
                height=args.height,
                seed=args.seed,
                references=args.reference,
                backend=args.backend,
                dit_mode=args.dit_mode,
                precision=args.precision,
                qwen=QwenOptions(**fields) if args.model != "h3" else None,
                on_progress=progress,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=args.output.parent, delete=False
            ) as tmp:
                try:
                    result.image.save(tmp, format="PNG")
                    tmp.flush()
                    os.replace(tmp.name, args.output)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            if args.metrics:
                args.metrics.parent.mkdir(parents=True, exist_ok=True)
                args.metrics.write_text(json.dumps(result.metrics, indent=2))
            print(args.output)
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
