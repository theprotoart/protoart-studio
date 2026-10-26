"""Model → LoRA: pull out what a fine-tuned checkpoint learned, as a small LoRA file.

A fine-tuned model is its base model plus changes. The change in each attention / feed-forward layer of the UNet
(and of the first text encoder) is worked out as (fine-tuned − base) and squeezed into a low-rank pair
(down × up, by SVD), the same way kohya's "extract LoRA from models" does. The result is a standard kohya-format
LoRA (diffusers layer names) that works in this app and in A1111, ComfyUI, Draw Things…

Both models must be checkpoints of the same type: .safetensors, or DiffusionBee .tdict models read directly (no
.safetensors copy needed first). Both (SD 1.5 with SD 1.5, SDXL with SDXL). The base should
be the model the fine-tune was made from (SD 1.5 base for most SD 1.5 models); with another base the LoRA holds the
difference between those two models instead. Weights are read one at a time, so it works on an 8 GB Mac. Rank 16–64:
higher keeps more of the model's look (bigger file).
"""
import json
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
# and the resnet layers (LoCon): a trained person or style changes these about as much as the attention layers
RES_LAYER = re.compile(r"(input_blocks\.\d+\.0|middle_block\.[02]|output_blocks\.\d+\.0)\.(in_layers\.2|out_layers\.3|emb_layers\.1|skip_connection)\.weight$")
_RES_PART = {"in_layers.2": "conv1", "out_layers.3": "conv2", "emb_layers.1": "time_emb_proj", "skip_connection": "conv_shortcut"}


def res_kohya_name(key):
    """'model.diffusion_model.input_blocks.1.0.in_layers.2.weight' -> 'lora_unet_down_blocks_0_resnets_0_conv1'."""
    k = key[len(UNET):-len(".weight")]
    m = re.match(r"input_blocks\.(\d+)\.0\.(.+)$", k)
    if m:
        i = int(m.group(1))
        if i == 0 or i % 3 == 0:
            return None  # conv_in and the downsamplers
        block = f"down_blocks.{(i - 1) // 3}.resnets.{(i - 1) % 3}"
    else:
        m = re.match(r"output_blocks\.(\d+)\.0\.(.+)$", k)
        if m:
            i = int(m.group(1))
            block = f"up_blocks.{i // 3}.resnets.{i % 3}"
        else:
            m = re.match(r"middle_block\.([02])\.(.+)$", k)
            if not m:
                return None
            block = f"mid_block.resnets.{int(m.group(1)) // 2}"
    part = _RES_PART.get(m.group(2))
    return ("lora_unet_" + f"{block}.{part}".replace(".", "_")) if part else None


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
    _low_rank.energy = (float(S.pow(2).sum()), float(S[:r].pow(2).sum()))  # all of the change, and what's kept
    return up.clamp(-hi, hi), down.clamp(-hi, hi)


class _TDictWeights:
    """A DiffusionBee .tdict model read like a .safetensors file (keys / get_tensor), one weight at a time."""
    def __init__(self, path):
        import tdict_reader
        info = tdict_reader.describe(path)
        if info.get("kind") != "checkpoint":
            raise ValueError(f"{Path(path).name} is a DiffusionBee {info.get('kind') or 'file'}, not a regular model: "
                             "only regular models can be turned into a LoRA (a DiffusionBee LoRA only needs "
                             "converting: Convert → Copy).")
        self.td = tdict_reader.TDict(path)
        if any(str(v.get("dtype", "")).startswith("cus") for v in self.td.keys.values()):
            raise ValueError(f"{Path(path).name} uses DiffusionBee's own compressed format, which can't be read.")
        self._keys = [k for k in self.td.keys if k.startswith(tdict_reader.CKPT_PREFIXES) and "_split_" not in k
                      and not k.endswith(".bias_by_weight")]

    def keys(self):
        return list(self._keys)

    def get_tensor(self, k):
        import numpy as np
        import torch
        return torch.from_numpy(np.array(self.td.array(k), dtype=np.float32))


def open_weights(path):
    """A model's weights, from a .safetensors or a DiffusionBee .tdict file."""
    p = str(path or "")
    if not p or not os.path.isfile(p):
        raise ValueError(f"{Path(p).name or 'A model'} wasn't found (is its drive connected?).")
    if p.lower().endswith(".safetensors"):
        from safetensors import safe_open
        return safe_open(p, "pt")
    if p.lower().endswith(".tdict"):
        return _TDictWeights(p)
    raise ValueError(f"Pick .safetensors or DiffusionBee .tdict models ({Path(p).name} isn't one).")


def extract(model, base, rank=32, name=None, text_encoder=True, emit=None, cancel_event=None, resnets=True, conv_rank=None):
    import torch
    from safetensors.torch import save_file
    emit = emit or (lambda m: None)
    rank = int(max(4, min(128, rank)))
    if model and base and os.path.realpath(model) == os.path.realpath(base):
        raise ValueError("Pick two different models: the one to turn into a LoRA, and the base it was made from.")
    fa, fb = open_weights(model), open_weights(base)
    keys_b = set(fb.keys())
    targets = []
    for k in fa.keys():
        if k.startswith(UNET) and UNET_LAYER.search(k):
            n = unet_kohya_name(k)
        elif text_encoder and TE_LAYER.search(k):
            n = te_kohya_name(k)
        elif resnets and k.startswith(UNET) and RES_LAYER.search(k):
            n = res_kohya_name(k)
        else:
            continue
        if n and k in keys_b:
            targets.append((k, n))
    if not targets:
        raise ValueError("These don't look like SD 1.5 / SDXL checkpoints (.safetensors or DiffusionBee .tdict files), or they aren't "
                         "the same type. Both must be SD 1.5, or both SDXL.")
    unet_a = sum(1 for k in fa.keys() if k.startswith(UNET) and UNET_LAYER.search(k))
    if sum(1 for k, _ in targets if k.startswith(UNET) and UNET_LAYER.search(k)) < 0.9 * unet_a:
        raise ValueError("These two models aren't the same type (SD 1.5 goes with SD 1.5, SDXL with SDXL).")
    stem = Path(model).stem
    if str(model).lower().endswith(".tdict"):
        import tdict_reader
        stem = tdict_reader.clean_name(model)  # DiffusionBee's file names carry extra bits: the model's own name
    out_name = (name or f"{stem}-lora-r{rank}").strip().replace("/", "-")[:120]
    paths.LORAS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.LORAS_DIR / (out_name + ".safetensors")
    n = 1
    while out.exists():
        out = paths.LORAS_DIR / f"{out_name}-{n}.safetensors"
        n += 1
    sd, kept, same = {}, 0, 0
    e_all = e_kept = 0.0  # how much of the change (in the layers a LoRA covers) the LoRA keeps
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
        kshape = tuple(diff.shape[1:]) if conv else None  # (in, kh, kw) of a conv layer
        if conv:
            diff = diff.reshape(diff.shape[0], -1)  # a 3x3 conv (resnets, LoCon): out x (in·3·3)
        if diff.abs().max() < 1e-6:
            same += 1
            continue
        up, down = _low_rank(diff, (conv_rank or rank) if (conv and kshape[1:] != (1, 1)) else rank)
        e_all += _low_rank.energy[0]
        e_kept += _low_rank.energy[1]
        if conv:  # down: a conv with the layer's own kernel, up: a 1x1 conv (kohya / LyCORIS LoCon layout)
            up, down = up[:, :, None, None], down.reshape(down.shape[0], *kshape)
        r = down.shape[0]
        sd[f"{kn}.lora_up.weight"] = up.contiguous().half()
        sd[f"{kn}.lora_down.weight"] = down.contiguous().half()
        sd[f"{kn}.alpha"] = torch.tensor(float(r)).half()
        kept += 1
    if not kept:
        raise ValueError("These two models are the same in every layer a LoRA covers: there's nothing to extract. "
                         "Pick the base model the other one was made from.")
    # and how much of the model's whole change is in layers a LoRA can't hold (resnets, norms…): a lot means the
    # base is most likely not the model it was made from
    covered = {k for k, _ in targets}
    e_other = 0.0
    kb = set(fb.keys())
    for i, k in enumerate(fa.keys()):
        if k in covered or not k.endswith(".weight") or k not in kb or not (k.startswith(UNET) or TE_LAYER.search(k)):
            continue
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError()
        a, b = fa.get_tensor(k).float(), fb.get_tensor(k).float()
        if a.shape == b.shape:
            e_other += float((a - b).pow(2).sum())
    kept_pct = round(100 * e_kept / e_all, 1) if e_all else 100.0
    covered_pct = round(100 * e_all / (e_all + e_other), 1) if (e_all + e_other) else 100.0
    fam = "sdxl" if any(k.startswith("conditioner.") for k in fa.keys()) else "sd15"
    meta = {"ss_network_module": "networks.lora", "ss_network_dim": str(rank), "ss_network_alpha": str(rank),
            **({"ss_network_args": json.dumps({"conv_dim": str(conv_rank or rank), "conv_alpha": str(conv_rank or rank)})} if resnets else {}),
            "ss_base_model_version": "sdxl_base_v1-0" if fam == "sdxl" else "sd_v1",
            "protoart_extracted_from": Path(model).name, "protoart_base": Path(base).name}
    save_file(sd, str(out), metadata=meta)
    return {"path": str(out), "name": out.stem, "layers": kept, "unchanged": same, "rank": rank, "family": fam,
            "mb": round(os.path.getsize(out) / 1e6, 1), "kept_pct": kept_pct, "covered_pct": covered_pct}


# ------------------------------------------------------------------ which model was it made from?
def _sample_keys(keys, n=80):
    """Weights spread over the whole model (UNet attention and resnets, and the text encoder)."""
    ks = sorted(k for k in keys if k.endswith(".weight") and (k.startswith(UNET) or "text_model" in k or "cond_stage_model" in k
                                                               or k.startswith("conditioner.")))
    return ks[:: max(1, len(ks) // n)][:n]


def change_from(model_w, base_path, keys):
    """How far a model is from a possible base: the size of the difference over the sampled weights, as a share
    of the base's own size (0.0 = the same model). None if they aren't the same kind of model."""
    fb = open_weights(base_path)
    kb = set(fb.keys())
    num = den = 0.0
    used = 0
    for k in keys:
        if k not in kb:
            continue
        a, b = model_w.get_tensor(k).float(), fb.get_tensor(k).float()
        if a.shape != b.shape:
            continue
        num += float((a - b).pow(2).sum())
        den += float(b.pow(2).sum())
        used += 1
    if used < max(5, len(keys) // 2) or den <= 0:
        return None
    return (num / den) ** 0.5


def find_base(model, candidates, progress=None, cancelled=None):
    """Rank the candidates by how close the model is to each: the one it was made from is far closer than the
    rest. -> [{"path", "change"}] closest first (change: 0.012 = 1.2 % different)."""
    fa = open_weights(model)
    keys = _sample_keys(fa.keys())
    out = []
    cands = [c for c in dict.fromkeys(candidates) if c and os.path.realpath(c) != os.path.realpath(model)]
    for i, c in enumerate(cands):
        if cancelled and cancelled():
            raise InterruptedError()
        if progress:
            progress(i, len(cands), c)
        try:
            ch = change_from(fa, c, keys)
        except Exception as e:  # a file that can't be read is just left out
            print("[lora_extract] couldn't compare with", Path(c).name, e, flush=True)
            ch = None
        if ch is not None:
            out.append({"path": c, "change": round(ch, 5)})
    out.sort(key=lambda r: r["change"])
    return out


def models_in_folders(folders, family=None, limit=300):
    """Full models (not LoRAs) in folders and their sub-folders, e.g. a whole external drive: .safetensors
    checkpoints and DiffusionBee .tdict models. family: keep only "sd15" / "sdxl" ones."""
    import model_hub
    import tdict_reader
    out, seen = [], set()
    for d in folders:
        root = Path(d)
        if not root.is_dir():
            continue
        try:
            files = sorted(root.rglob("*.safetensors"))
        except OSError:
            files = []
        for f in files:
            if len(out) >= limit:
                return out
            if f.name.startswith("._") or str(f) in seen:
                continue
            try:
                if f.stat().st_size < 1_500_000_000:  # LoRAs, embeddings, VAEs: not full models
                    continue
                info = model_hub.inspect_local(f)
            except Exception:
                continue
            if family and info.get("family") != family:
                continue
            seen.add(str(f))
            out.append(str(f))
        try:
            for t in tdict_reader.scan([str(root)], recursive=True):
                if t.get("kind") == "checkpoint" and (not family or t.get("family") == family) and t["file"] not in seen:
                    seen.add(t["file"])
                    out.append(t["file"])
        except Exception:
            pass
    return out[:limit]


def verdict(ranked):
    """What the ranking means. "match": the closest is clearly the base (a model is about 1.4x further from a
    sister than from their shared base). "sisters": several are about equally close: they and this model were
    made from the same base, which isn't among them. "none": nothing is close."""
    if not ranked:
        return "none"
    best = ranked[0]["change"]
    if best >= 0.12:
        return "none"
    if len(ranked) == 1 or ranked[1]["change"] >= best * 1.3:
        return "match"
    return "sisters"
