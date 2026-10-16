"""Read DiffusionBee's .tdict files directly, so models, LoRAs and embeddings you made or
imported in DiffusionBee work here without converting or copying anything.

Independent implementation written for Protoart Studio (Apache-2.0) so users can open files they
already have (interoperability). It contains no DiffusionBee code; only the file layout below, which
was worked out from DiffusionBee's published format, is shared.

TDict layout:
  0..64     file header
  64..128   block header: uint64 [magic 8030895855, length, data_start, ...]
  128..288  "extra head": 20 x uint64
              [0] magic 4346464   [2] format version   [3] ctdict id (what kind of model)
              [4]/[5] start/end of the weights JSON    [6]/[7] start/end of the metadata JSON
  weights JSON: {key: {start, end, shape, dtype}} -> raw little-endian arrays in the file

Only the JSON header is read when scanning; tensors are memory-mapped on use.
"""
import json
import re
import struct
from pathlib import Path

import numpy as np

BLOCK_MAGIC = 8030895855
EXTRA_MAGIC = 4346464

# ctdict ids written by DiffusionBee
KIND_BY_ID = {
    12: ("checkpoint", "sd15"), 1012: ("checkpoint", "sd15"),
    13: ("inpaint", "sd15"), 1013: ("inpaint", "sd15"),
    15: ("checkpoint", "sd2"), 1015: ("checkpoint", "sd2"),
    1014: ("controlnet", "sd15"),
    3031: ("checkpoint", "sdxl_f8"),
    69: ("lora", None),
    87: ("embedding", None),
}
DTYPES = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int64": np.int64,
          "uint8": np.uint8, "int8": np.int8}
CKPT_PREFIXES = ("model.diffusion_model.", "first_stage_model.", "cond_stage_model.", "conditioner.")


class TDictError(ValueError):
    pass


def read_header(path):
    """Return {"ctdict": id, "keys": {...}, "meta": {...}} without reading tensor data."""
    with open(path, "rb") as f:
        head = f.read(288)
        if len(head) < 288:
            raise TDictError("file too small")
        block = struct.unpack("<8Q", head[64:128])
        extra = struct.unpack("<20Q", head[128:288])
        if block[0] != BLOCK_MAGIC or extra[0] != EXTRA_MAGIC:
            raise TDictError("not a DiffusionBee .tdict file")
        ws, we, ms, me = extra[4], extra[5], extra[6], extra[7]
        if not (0 < ws < we) or we - ws > 200_000_000:
            raise TDictError("bad weights table")
        f.seek(ws)
        keys = json.loads(f.read(we - ws))
        meta = {}
        if 0 < ms < me and me - ms < 10_000_000:
            f.seek(ms)
            try:
                meta = json.loads(f.read(me - ms) or b"{}")
            except ValueError:
                meta = {}
    return {"ctdict": int(extra[3]), "version": int(extra[2]), "keys": keys, "meta": meta}


def describe(path, header=None):
    """Work out what a tdict holds: {kind, family, name, ...}. kind is one of
    checkpoint / lora / embedding / inpaint / controlnet / unknown."""
    h = header or read_header(path)
    kind, family = KIND_BY_ID.get(h["ctdict"], ("unknown", None))
    keys = h["keys"]
    custom = any(str(v.get("dtype", "")).startswith("cus") for v in keys.values())
    if kind == "unknown":  # older files: guess from the key names
        ks = list(keys)
        if any(k.endswith("_lora_up") for k in ks):
            kind = "lora"
        elif ks == ["emb_params"] or "emb_params" in keys:
            kind = "embedding"
        elif any(k.startswith("model.diffusion_model.") for k in ks):
            kind = "checkpoint"
    if kind == "lora":
        # DiffusionBee makes SD 1.5 LoRAs unless the weights clearly say SDXL: so "can't tell" means SD 1.5
        family = lora_family(keys)
        if family == "unknown":
            family = "sd15"
    elif kind == "embedding":
        shape = (keys.get("emb_params") or {}).get("shape") or [0, 0]
        family = {768: "sd15", 1024: "sd2"}.get(shape[-1], "unknown")
    elif kind == "checkpoint" and family is None:
        family = "sdxl" if any(k.startswith("conditioner.embedders.1") for k in keys) else "sd15"
    supported = (not custom) and (
        (kind == "checkpoint" and family == "sd15") or
        (kind == "lora" and family in ("sd15", "sdxl")) or
        (kind == "embedding" and family == "sd15"))
    return {"kind": kind, "family": family or "unknown", "name": clean_name(path),
            "supported": supported, "ctdict": h["ctdict"]}


def clean_name(path):
    """DiffusionBee saves imports as '<name>_<name>.tdict'; show just '<name>'."""
    stem = Path(path).stem
    n = len(stem)
    if n % 2 == 1 and stem[n // 2] == "_" and stem[: n // 2] == stem[n // 2 + 1:]:
        return stem[: n // 2]
    return stem


def lora_family(keys):
    ks = list(keys)
    if any(k.startswith("conditioner.embedders.1") for k in ks):
        return "sdxl"
    for k, v in keys.items():
        # SD 1.x: first transformer block in input_blocks.1 has 320 channels and conv proj_in;
        # SDXL has no attention in input_blocks.1 and uses 640/1280 channels.
        if "input_blocks.1.1." in k:
            return "sd15"
        if re.search(r"input_blocks\.[45]\.1\.transformer_blocks\.[1-9]", k):
            return "sdxl"
    joined = "\n".join(ks)
    if "label_emb" in joined or "transformer_blocks.9" in joined:
        return "sdxl"
    return "sd15" if "model.diffusion_model." in joined else "unknown"


class TDict:
    def __init__(self, path):
        self.path = str(path)
        self.header = read_header(path)
        self.keys = self.header["keys"]
        self._mm = None

    def array(self, key):
        info = self.keys[key]
        dt = DTYPES.get(info["dtype"])
        if dt is None:
            raise TDictError(f"unsupported data type {info['dtype']} for {key}")
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        raw = self._mm[info["start"]:info["end"]]
        return np.frombuffer(raw, dtype=dt).reshape(info["shape"])

    def tensor(self, key, dtype=None):
        import torch
        t = torch.from_numpy(np.array(self.array(key)))  # copy: memmap is read-only
        return t.to(dtype) if dtype is not None else t

    # -------------------------------------------------------------- conversions
    def checkpoint_keys(self):
        """Original-format (LDM) weight names, minus the helper arrays DiffusionBee adds
        (split copies, bias/weight ratios, zeros/ones, timestep tables, causal mask...)."""
        return [k for k in self.keys
                if k.startswith(CKPT_PREFIXES) and "_split_" not in k and not k.endswith(".bias_by_weight")]

    def checkpoint_state_dict(self, dtype=None):
        return {k: self.tensor(k, dtype) for k in self.checkpoint_keys()}

    def lora_arrays(self):
        """Convert to the common Kohya LoRA layout (numpy) that diffusers' load_lora_weights understands.
        DiffusionBee stores delta = up @ down * scale; Kohya uses alpha / rank, so alpha = scale * rank."""
        out = {}
        roots = sorted({k[: -len("_lora_up")] for k in self.keys if k.endswith("_lora_up")})
        for root in roots:
            name = kohya_name(root)
            if name is None or root + "_lora_down" not in self.keys:
                continue
            up = np.asarray(self.array(root + "_lora_up"), dtype=np.float32)
            down = np.asarray(self.array(root + "_lora_down"), dtype=np.float32)
            scale_key = root + "_lora_scale"
            scale = float(self.array(scale_key).reshape(-1)[0]) if scale_key in self.keys else 1.0
            out[name + ".lora_up.weight"] = up
            out[name + ".lora_down.weight"] = down
            out[name + ".alpha"] = np.array(scale * down.shape[0], dtype=np.float32)
        if not out:
            raise TDictError("no LoRA weights found in this file")
        return out

    def lora_state_dict(self):
        import torch
        return {k: torch.from_numpy(np.array(v)) for k, v in self.lora_arrays().items()}

    def embedding(self):
        import torch
        return self.tensor("emb_params", torch.float32)


RESNET_PARTS = {"in_layers.2": "conv1", "out_layers.3": "conv2", "emb_layers.1": "time_emb_proj",
                "skip_connection": "conv_shortcut", "in_layers.0": "norm1", "out_layers.0": "norm2"}


def unet_ldm_to_diffusers(path, layers_per_block=2):
    """Translate an original (LDM) UNet module path to the diffusers module path, e.g.
    'input_blocks.4.1.proj_in' -> 'down_blocks.1.attentions.0.proj_in'. Returns None if unknown.
    (diffusers only does this remapping itself for SDXL, so SD 1.5 LoRAs need it done here.)"""
    m = re.match(r"(input_blocks|output_blocks)\.(\d+)\.(\d+)\.?(.*)$", path)
    n = layers_per_block + 1
    if m:
        kind, i, j, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if kind == "input_blocks":
            if i == 0:
                return "conv_in" if not rest else None
            block, layer = (i - 1) // n, (i - 1) % n
            if layer == layers_per_block:  # the downsampler slot
                return f"down_blocks.{block}.downsamplers.0.conv" if rest == "op" else None
            head = f"down_blocks.{block}." + ("resnets" if j == 0 else "attentions") + f".{layer}"
        else:
            block, layer = i // n, i % n
            if j == 0:
                head = f"up_blocks.{block}.resnets.{layer}"
            elif j == 1 and not rest.startswith("conv"):
                head = f"up_blocks.{block}.attentions.{layer}"
            else:  # upsampler (index 1 or 2 depending on whether the block has attention)
                return f"up_blocks.{block}.upsamplers.0.conv" if rest == "conv" else None
        if j == 0:
            rest = RESNET_PARTS.get(rest, rest)
        return head + ("." + rest if rest else "")
    m = re.match(r"middle_block\.(\d)\.(.*)$", path)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        if i == 1:
            return "mid_block.attentions.0." + rest
        return f"mid_block.resnets.{i // 2}." + RESNET_PARTS.get(rest, rest)
    simple = {"time_embed.0": "time_embedding.linear_1", "time_embed.2": "time_embedding.linear_2",
              "out.2": "conv_out", "label_emb.0.0": "add_embedding.linear_1", "label_emb.0.2": "add_embedding.linear_2"}
    return simple.get(path)


def kohya_name(ldm_key):
    """'model.diffusion_model.input_blocks.1.1.proj_in.weight' -> 'lora_unet_down_blocks_0_attentions_0_proj_in'."""
    k = ldm_key
    if "_split_" in k:
        return None  # DiffusionBee also stores split copies of some layers; the full layer is there too
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    if k.startswith("model.diffusion_model."):
        d = unet_ldm_to_diffusers(k[len("model.diffusion_model."):])
        return "lora_unet_" + d.replace(".", "_") if d else None
    for prefix, kp in (("cond_stage_model.transformer.", "lora_te_"),
                       ("conditioner.embedders.0.transformer.", "lora_te1_")):
        if k.startswith(prefix):
            return kp + k[len(prefix):].replace(".", "_")
    return None  # e.g. SDXL's second (OpenCLIP) text encoder: not mapped


# ------------------------------------------------------------------ scanning
_cache = {}  # path -> (mtime, size, description)


def scan(dirs, recursive=False):
    """Describe every .tdict in the given folders (and their sub-folders when recursive).
    Header reads are cached by mtime/size."""
    out = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.tdict") if recursive else d.glob("*.tdict")):
            if p.name.startswith("._"):
                continue  # macOS resource-fork files on external drives
            try:
                st = p.stat()
                c = _cache.get(str(p))
                if c and c[0] == st.st_mtime and c[1] == st.st_size:
                    info = c[2]
                else:
                    info = describe(p)
                    _cache[str(p)] = (st.st_mtime, st.st_size, info)
            except Exception as e:
                info = {"kind": "unknown", "family": "unknown", "name": clean_name(p),
                        "supported": False, "error": str(e)}
                st = None
            out.append(dict(info, file=str(p), size_mb=round(st.st_size / 1e6, 1) if st else 0))
    return out


# ------------------------------------------------------------------ conversion to .safetensors
CONVERTIBLE = ("lora", "checkpoint", "embedding", "controlnet")


def convert_to_safetensors(path, out_dir, progress=None):
    """Write a standard .safetensors copy of a DiffusionBee .tdict. Uses numpy only (no GPU).
      lora       -> Kohya layout (A1111, ComfyUI, Draw Things, Protoart, diffusers)
      checkpoint -> original SD layout (A1111, ComfyUI, Draw Things, diffusers from_single_file)
      embedding  -> A1111 textual-inversion layout ({"emb_params": ...})
      controlnet -> original ControlNet layout
    Returns the new file's path. The original .tdict is left untouched."""
    info = describe(path)
    kind = info["kind"]
    if kind not in CONVERTIBLE:
        raise TDictError(f"{info['name']}: {kind} files can't be converted")
    td = TDict(path)
    if any(str(v.get("dtype", "")).startswith("cus") for v in td.keys.values()):
        raise TDictError(f"{info['name']}: uses DiffusionBee's own compressed format, which can't be converted")
    meta = {"converted_from": "DiffusionBee .tdict", "protoart_kind": kind, "beestudio_family": info.get("family") or ""}
    if kind == "lora":
        tensors = td.lora_arrays()
        meta.update({"ss_network_module": "networks.lora",
                     "ss_base_model_version": "sdxl_base_v1-0" if info["family"] == "sdxl" else "sd_v1"})
    elif kind == "embedding":
        tensors = {"emb_params": np.ascontiguousarray(td.array("emb_params"))}
    else:
        keys = [k for k in td.keys if "_split_" not in k and not k.endswith(".bias_by_weight")
                and not k.startswith(("zeros_", "ones_", "temb_coefficients", "causal_mask", "aux_output_conv",
                                      "alphas_cumprod"))]
        if kind == "checkpoint":
            keys = [k for k in keys if k.startswith(CKPT_PREFIXES)]
        tensors = {}
        for i, k in enumerate(keys):
            tensors[k] = np.ascontiguousarray(td.array(k))
            if progress and i % 50 == 0:
                progress(i, len(keys))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{info['name']}.safetensors"
    n = 2
    while out.exists():
        out = out_dir / f"{info['name']}-{n}.safetensors"
        n += 1
    tmp = out.with_suffix(".tmp")
    save_safetensors(tensors, tmp, meta)
    tmp.replace(out)
    return str(out)


_ST_DTYPES = {np.dtype("float16"): "F16", np.dtype("float32"): "F32", np.dtype("float64"): "F64",
              np.dtype("int64"): "I64", np.dtype("int32"): "I32", np.dtype("int8"): "I8", np.dtype("uint8"): "U8"}


def save_safetensors(tensors, path, metadata=None):
    """Minimal .safetensors writer (the format is a JSON header followed by raw little-endian data)."""
    header, offset, order = {}, 0, []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        if a.dtype not in _ST_DTYPES:
            a = a.astype(np.float32)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        n = a.nbytes
        header[name] = {"dtype": _ST_DTYPES[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + n]}
        order.append(a)
        offset += n
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for a in order:
            f.write(a.tobytes())
