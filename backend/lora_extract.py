"""Model → LoRA: pull out what a fine-tuned checkpoint learned, as a small LoRA file.

A fine-tuned model is its base model plus changes. The change in each attention / feed-forward layer of the UNet
(and of the first text encoder) is worked out as (fine-tuned − base) and squeezed into a low-rank pair
(down × up, by SVD), the same way kohya's "extract LoRA from models" does. The result is a standard kohya-format
LoRA (diffusers layer names) that works in this app and in A1111, ComfyUI, Draw Things…

Both models must be .safetensors checkpoints of the same type (SD 1.5 with SD 1.5, SDXL with SDXL). The base should
be the model the fine-tune was made from (SD 1.5 base for most SD 1.5 models); with another base the LoRA holds the
difference between those two models instead. Weights are read one at a time, so it works on an 8 GB Mac. Rank 16–64:
higher keeps more of the model's look (bigger file).
"""
import os
import re
from pathlib import Path

import paths

UNET = "model.diffusion_model."
TE_PREFIXES = {  # checkpoint key prefix of the (first) CLIP text encoder -> kohya prefix
    "cond_stage_model.transformer.text_model.": "lora_te_text_model_",          # SD 1.5
    "conditioner.embedders.0.transformer.text_model.": "lora_te1_text_model_",  # SDXL (CLIP-L)
}
# the layers a LoRA usually covers
UNET_LAYER = re.compile(r"(attn[12]\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)|proj_in|proj_out)\.weight$")
TE_LAYER = re.compile(r"encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))\.weight$")


def unet_kohya_name(key):
    """'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight' ->
    'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q' (diffusers names, like kohya)."""
    k = key[len(UNET):-len(".weight")]
    m = re.match(r"input_blocks\.(\d+)\.1\.(.+)$", k)
    if m:
        i = int(m.group(1))
        name = f"down_blocks.{(i - 1) // 3}.attentions.{(i - 1) % 3}.{m.group(2)}"
    else:
        m = re.match(r"output_blocks\.(\d+)\.1\.(.+)$", k)
        if m:
            i = int(m.group(1))
            name = f"up_blocks.{i // 3}.attentions.{i % 3}.{m.group(2)}"
        else:
            m = re.match(r"middle_block\.1\.(.+)$", k)
            if not m:
                return None
            name = f"mid_block.attentions.0.{m.group(1)}"
    return "lora_unet_" + name.replace(".", "_")


def te_kohya_name(key):
    for pre, kp in TE_PREFIXES.items():
        if key.startswith(pre):
            return kp + key[len(pre):-len(".weight")].replace(".", "_")
    return None


def _low_rank(diff, rank):
    """diff (out, in) -> (up (out, r), down (r, in)). The singular values are shared evenly between the two halves
    (√S each), so both stay in a similar range for half precision; only extreme outliers (far beyond the 99th
    percentile) are clipped, so the LoRA keeps the change faithfully."""
    import torch
    r = min(rank, diff.shape[0], diff.shape[1])
    U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
    s = torch.sqrt(S[:r])
    up, down = U[:, :r] * s[None, :], s[:, None] * Vh[:r, :]
    dist = torch.cat([up.flatten(), down.flatten()]).abs()
    hi = torch.quantile(dist[:: max(1, dist.numel() // 1_000_000)], 0.99) * 8
    return up.clamp(-hi, hi), down.clamp(-hi, hi)


def extract(model, base, rank=32, name=None, text_encoder=True, emit=None, cancel_event=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    emit = emit or (lambda m: None)
    rank = int(max(4, min(128, rank)))
    for f in (model, base):
        if not f or not str(f).endswith(".safetensors") or not os.path.isfile(f):
            raise ValueError(f"Pick .safetensors checkpoint files ({Path(str(f or '')).name or 'missing'} isn't one).")
    if os.path.realpath(model) == os.path.realpath(base):
        raise ValueError("Pick two different models: the one to turn into a LoRA, and the base it was made from.")
    fa, fb = safe_open(model, "pt"), safe_open(base, "pt")
    keys_b = set(fb.keys())
    targets = []
    for k in fa.keys():
        if k.startswith(UNET) and UNET_LAYER.search(k):
            n = unet_kohya_name(k)
        elif text_encoder and TE_LAYER.search(k):
            n = te_kohya_name(k)
        else:
            continue
        if n and k in keys_b:
            targets.append((k, n))
    if not targets:
        raise ValueError("These don't look like SD 1.5 / SDXL checkpoints (single .safetensors files), or they aren't "
                         "the same type. Both must be SD 1.5, or both SDXL.")
    unet_a = sum(1 for k in fa.keys() if k.startswith(UNET) and UNET_LAYER.search(k))
    if sum(1 for k, _ in targets if k.startswith(UNET)) < 0.9 * unet_a:
        raise ValueError("These two models aren't the same type (SD 1.5 goes with SD 1.5, SDXL with SDXL).")
    out_name = (name or f"{Path(model).stem}-lora-r{rank}").strip().replace("/", "-")[:120]
    paths.LORAS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.LORAS_DIR / (out_name + ".safetensors")
    n = 1
    while out.exists():
        out = paths.LORAS_DIR / f"{out_name}-{n}.safetensors"
        n += 1
    sd, kept, same = {}, 0, 0
    for i, (k, kn) in enumerate(targets):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError()
        if i % 20 == 0:
            emit({"type": "progress", "image": 1, "images": 1, "step": i + 1, "steps": len(targets)})
            emit({"type": "status", "msg": f"Model → LoRA: layer {i + 1} of {len(targets)}…"})
        a, b = fa.get_tensor(k).float(), fb.get_tensor(k).float()
        if a.shape != b.shape:
            continue
        diff = a - b
        del a, b
        conv = diff.dim() == 4
        if conv:
            if diff.shape[2:] != (1, 1):
                continue
            diff = diff[:, :, 0, 0]
        if diff.abs().max() < 1e-6:
            same += 1
            continue
        up, down = _low_rank(diff, rank)
        if conv:
            up, down = up[:, :, None, None], down[:, :, None, None]
        r = down.shape[0]
        sd[f"{kn}.lora_up.weight"] = up.contiguous().half()
        sd[f"{kn}.lora_down.weight"] = down.contiguous().half()
        sd[f"{kn}.alpha"] = torch.tensor(float(r)).half()
        kept += 1
    if not kept:
        raise ValueError("These two models are the same in every layer a LoRA covers: there's nothing to extract. "
                         "Pick the base model the other one was made from.")
    fam = "sdxl" if any(k.startswith("conditioner.") for k in fa.keys()) else "sd15"
    meta = {"ss_network_module": "networks.lora", "ss_network_dim": str(rank), "ss_network_alpha": str(rank),
            "ss_base_model_version": "sdxl_base_v1-0" if fam == "sdxl" else "sd_v1",
            "protoart_extracted_from": Path(model).name, "protoart_base": Path(base).name}
    save_file(sd, str(out), metadata=meta)
    return {"path": str(out), "name": out.stem, "layers": kept, "unchanged": same, "rank": rank, "family": fam,
            "mb": round(os.path.getsize(out) / 1e6, 1)}
