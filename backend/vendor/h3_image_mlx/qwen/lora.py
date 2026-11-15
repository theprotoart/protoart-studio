"""Qwen adapter validation, shared with the desktop model contract."""

import math
import re

from safetensors import safe_open

MODULES = {
    "attn.to_q": (4096, 4096),
    "attn.to_k": (4096, 4096),
    "attn.to_v": (4096, 4096),
    "attn.to_out": (4096, 4096),
    "attn.to_out.0": (4096, 4096),
    "img_mlp.gate_layer": (4096, 12288),
    "img_mlp.proj": (4096, 12288),
    "img_mlp.gate_up": (4096, 24576),
    "img_mlp.out": (12288, 4096),
}


def inspect_adapter(path):
    pairs = {}
    alpha = set()
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            name = re.sub(
                r"^(?:base_model\.model\.)?(?:transformer\.|diffusion_model\.)?",
                "",
                key,
            )
            module, sep, suffix = name.partition(".lora")
            if not sep and name.endswith(".alpha"):
                module, suffix = name[:-6], "alpha"
            flat = re.fullmatch(
                r"lora_(?:unet|transformer)_transformer_blocks_(\d+)_(.+)", module
            )
            if flat:
                names = {k.replace(".", "_"): k for k in MODULES}
                module = f"transformer_blocks.{flat[1]}.{names.get(flat[2], '')}"
            m = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", module)
            if not m or int(m[1]) >= 32 or m[2] not in MODULES:
                raise ValueError(f"当前 Qwen-Image 2.1 引擎不能完整加载此层：{key}")
            tensor = f.get_slice(key)
            shape = tensor.get_shape()
            if tensor.get_dtype() not in ("F16", "BF16", "F32", "F64"):
                raise ValueError(f"LoRA 张量类型不支持：{key}")
            if suffix == "alpha" and math.prod(shape) == 1:
                alpha.add(module)
                continue
            suffix = suffix.replace(".default.weight", ".weight")
            role = {
                "_A.weight": "a",
                "_down.weight": "a",
                ".down.weight": "a",
                "_B.weight": "b",
                "_up.weight": "b",
                ".up.weight": "b",
            }.get(suffix)
            if role is None or len(shape) != 2 or min(shape) < 1:
                raise ValueError(f"无效的 LoRA 矩阵：{key}")
            dims = MODULES[m[2]]
            if (shape[1] if role == "a" else shape[0]) != dims[role == "b"]:
                raise ValueError(f"矩阵尺寸与 Qwen-Image 2.1 不匹配：{key}")
            pair = pairs.setdefault(module, {})
            if role in pair:
                raise ValueError(f"重复的 LoRA 矩阵：{key}")
            pair[role] = shape[0] if role == "a" else shape[1]
    if (
        not pairs
        or alpha - set(pairs)
        or any(p.get("a") != p.get("b") for p in pairs.values())
    ):
        raise ValueError("LoRA 缺少成对的上下投影矩阵")
    return len(pairs)
