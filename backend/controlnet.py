"""ControlNet for the Image to image tab (SD 1.5, and your own SDXL ControlNets).

A ControlNet makes the picture follow a "control map" made from your image: its edges (Canny),
depth, line art, a scribble, or the image itself (Tile, for adding detail / upscaling).

Models: your DiffusionBee ControlNets are used in place when you have them (offline); otherwise
the official ControlNet 1.1 model is downloaded once from Hugging Face. Your own ControlNet files
(~/.protoartstudio/controlnets, Civitai downloads go there) can be SD 1.5 or SDXL (Pony, Illustrious and
NoobAI use SDXL's): the file says which, and its size (full, mid or small) is read from the file too.
"""
import os
from pathlib import Path

import numpy as np

TYPES = {
    #  id: (label, Hugging Face repo, DiffusionBee file name hint, what the map is)
    "tile": ("Tile (add detail, keep the picture)", "lllyasviel/control_v11f1e_sd15_tile", "tile",
             "the image itself"),
    "canny": ("Canny edges (keep outlines)", "lllyasviel/control_v11p_sd15_canny", "canny", "edge lines"),
    "depth": ("Depth (keep the 3-D layout)", "lllyasviel/control_v11f1p_sd15_depth", "depth", "a depth map"),
    "lineart": ("Line art (keep the drawing)", "lllyasviel/control_v11p_sd15_lineart", "lineart", "line art"),
    "scribble": ("Scribble (from a rough sketch)", "lllyasviel/control_v11p_sd15_scribble", "scribble",
                 "the sketch's lines"),
    "softedge": ("Soft edges (looser outlines)", "lllyasviel/control_v11p_sd15_softedge", "softedge",
                 "soft edges"),
    "openpose": ("OpenPose (copy a person's pose)", "lllyasviel/control_v11p_sd15_openpose", "openpose",
                 "the pose skeleton"),
}
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
# Depth Anything V2 (https://github.com/DepthAnything/Depth-Anything-V2), Hugging Face ports.
#   size: (repo, download GB, licence)
DEPTH_MODELS = {
    "small": ("depth-anything/Depth-Anything-V2-Small-hf", 0.1, "Apache 2.0"),
    "base": ("depth-anything/Depth-Anything-V2-Base-hf", 0.4, "CC BY-NC 4.0 (non-commercial)"),
    "large": ("depth-anything/Depth-Anything-V2-Large-hf", 1.3, "CC BY-NC 4.0 (non-commercial)"),
}
SKIP_DBEE = set()   # kinds whose DiffusionBee file gave a broken (flat grey) result this session
LAST_SOURCE = {}    # kind -> "diffusionbee" | "official"
SAFE_KINDS = set()  # kinds whose numbers broke at half precision: run them at full precision (see keep_finite)


def keep_finite(model, out_dtype):
    """Run a ControlNet at full precision beside a half-precision model. Some ControlNets (many QR / pattern ones)
    give numbers too big for half precision on Apple GPUs; the picture then collapses to one flat colour. Inputs are
    turned to float32, and the result is cleaned (no NaN / infinity) and turned back to the model's precision."""
    import dataclasses
    import torch
    run = model.forward
    big = 6.0e4  # just under half precision's largest number

    def up(x):
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    def down(x):
        if torch.is_tensor(x) and x.is_floating_point():
            return torch.nan_to_num(x, nan=0.0, posinf=big, neginf=-big).clamp(-big, big).to(out_dtype)
        if dataclasses.is_dataclass(x) and not isinstance(x, type):  # diffusers' outputs (also dicts): keep the type
            for f in dataclasses.fields(x):
                setattr(x, f.name, down(getattr(x, f.name)))
            return x
        if isinstance(x, (list, tuple)):
            return type(x)(down(v) for v in x)
        if isinstance(x, dict):
            return {k: down(v) for k, v in x.items()}
        return x

    def forward(*args, **kwargs):
        args = [up(v) for v in args]
        kwargs = {k: ({kk: up(vv) for kk, vv in v.items()} if isinstance(v, dict) else up(v)) for k, v in kwargs.items()}
        return down(run(*args, **kwargs))

    model.forward = forward
    return model
DOWNLOAD_GB = 1.45


def list_types():
    found = _dbee_items()  # scan DiffusionBee's folders once, not once per type
    out = []
    for k, (label, repo, hint, what) in TYPES.items():
        out.append({"id": k, "label": label, "map": what, "diffusionbee": bool(_dbee_file(hint, found)), "family": "sd15"})
    for u in user_files():
        if u.get("problem"):
            continue  # listed in the Models tab with the reason; it can't be picked
        m = guess_map(u["name"])
        fam = {"sd15": "SD 1.5", "sdxl": "SDXL"}.get(u["family"], "")
        out.append({"id": "user:" + u["name"], "label": f"{u['name']} (your {fam} ControlNet)".replace("  ", " "),
                    "map": TYPES[m][3] if m in TYPES and m != "tile" else "your picture as it is",
                    "user": True, "size_mb": u["size_mb"], "family": u["family"], "makes_map": m})
    return out


# The kind of map a ControlNet file expects, from words in its name ("…canny…", "depth", "openpose"…):
# that map is made from your picture for it, as for the built-in ones. Anything else gets the picture as it is.
_MAP_WORDS = (("openpose", "openpose"), ("pose", "openpose"), ("canny", "canny"), ("depth", "depth"),
              ("lineart", "lineart"), ("line_art", "lineart"), ("line-art", "lineart"), ("anime_line", "lineart"),
              ("softedge", "softedge"), ("soft_edge", "softedge"), ("hed", "softedge"), ("pidi", "softedge"),
              ("scribble", "scribble"), ("sketch", "scribble"), ("tile", "tile"))


def guess_map(name):
    import re
    low = str(name).lower()
    for word, kind in _MAP_WORDS:
        if re.search(r"(^|[^a-z])" + re.escape(word), low):
            return kind
    return None


_INFO = {}  # file -> (mtime, size, info) so the folder isn't re-read every time the list is shown


def inspect_file(path):
    """What a ControlNet file is, from its weight names and shapes (nothing is loaded):
    {"family": "sd15"|"sdxl"|None, "layout": "diffusers"|"ldm", "problem": None or why it can't be used}."""
    try:
        st = os.stat(path)
    except OSError as e:
        return {"family": None, "layout": None, "problem": f"can't read the file ({e.strerror})"}
    hit = _INFO.get(path)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    info = _inspect(path)
    _INFO[path] = (st.st_mtime, st.st_size, info)
    return info


def _shapes(path):
    """Weight names and shapes from a .safetensors file's header (a few KB read, no weights)."""
    import json
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= n <= 100_000_000:
            raise ValueError("bad header")
        head = json.loads(f.read(n))
    return {k: tuple(v.get("shape") or ()) for k, v in head.items() if k != "__metadata__" and isinstance(v, dict)}


def _inspect(path):
    try:
        shapes = _shapes(path)
    except Exception as e:
        return {"family": None, "layout": None, "problem": f"not a readable .safetensors file ({e.__class__.__name__})"}
    keys = list(shapes)
    low = " ".join(keys[:4000]).lower()
    if "lllite" in low:
        return {"family": "sdxl", "layout": None, "problem": "a ControlNet-LLLite file (not supported yet)"}
    if "task_embedding" in shapes or any(k.startswith(("spatial_ch_projs", "control_add_embedding")) for k in keys):
        return {"family": "sdxl", "layout": None, "problem": "a ControlNet Union file (not supported yet)"}
    if any(k.startswith(("adapter.", "body.0.")) for k in keys) and not any("zero_conv" in k or "controlnet_" in k for k in keys):
        return {"family": None, "layout": None, "problem": "a T2I-Adapter, not a ControlNet (not supported yet)"}
    if any(".lora_" in k or "lora_controlnet" in k or k.endswith((".up", ".down")) for k in keys):
        return {"family": "sdxl", "layout": None, "problem": "a Control-LoRA file (not supported yet)"}
    if any(k.startswith(("controlnet_cond_embedding.", "controlnet_down_blocks.")) for k in keys):
        layout = "diffusers"
    elif any("input_hint_block" in k or "zero_convs" in k for k in keys):
        layout = "ldm"
    else:
        return {"family": None, "layout": None, "problem": "doesn't look like a ControlNet"}
    dims = {shp[1] for k, shp in shapes.items() if k.endswith("attn2.to_k.weight") and len(shp) == 2}
    if 2048 in dims or any(k.endswith(("add_embedding.linear_1.weight", "label_emb.0.0.weight")) for k in keys):
        fam = "sdxl"
    elif 768 in dims:
        fam = "sd15"
    elif 1024 in dims:
        return {"family": None, "layout": layout, "problem": "an SD 2 ControlNet (SD 2 isn't supported)"}
    elif not dims:
        fam = "sdxl" if any("label_emb" in k or "add_embedding" in k for k in keys) else "sd15"
    else:
        return {"family": None, "layout": layout, "problem": "made for a model type this app doesn't know"}
    return {"family": fam, "layout": layout, "problem": None}


def user_files():
    """Your own ControlNet files (~/.protoartstudio/controlnets, Civitai downloads go there), SD 1.5 or SDXL."""
    import paths
    d = paths.CONTROLNET_DIR
    out = []
    if d.is_dir():
        for p in sorted(d.rglob("*.safetensors")):
            if p.name.startswith("._"):
                continue
            try:
                size = round(p.stat().st_size / 1e6, 1)
            except OSError:
                continue
            info = inspect_file(str(p))
            out.append({"name": p.stem, "file": str(p), "size_mb": size, "family": info["family"],
                        "layout": info["layout"], "problem": info["problem"], "map": guess_map(p.stem)})
    return out


def family_of(kind):
    """The model type a ControlNet works with: "sd15" (the built-in ones), or what your file is made for."""
    if str(kind).startswith("user:"):
        name = kind[len("user:"):]
        return next((u["family"] for u in user_files() if u["name"] == name), None)
    return "sd15" if kind in TYPES else None


def is_known(kind):
    return kind in TYPES or (str(kind).startswith("user:") and any("user:" + u["name"] == kind and not u["problem"]
                                                                   for u in user_files()))


def _load_user(kind, unet, dtype, emit):
    """Your own ControlNet file, converted in place (no download): the same way DiffusionBee's are read."""
    import diffusers
    import torch
    from safetensors.torch import load_file
    name = kind[len("user:"):]
    u = next((u for u in user_files() if u["name"] == name), None)
    if not u:
        raise ValueError(f"the ControlNet file “{name}” isn't in the ControlNet folder any more")
    if u["problem"]:
        raise ValueError(f"“{name}” is {u['problem']}")
    f = u["file"]
    return _load_user_xl(kind, name, f, u["layout"], unet, dtype, emit, family=u["family"])


def _read_half(path, dtype=None):
    """A .safetensors file's weights, each turned into dtype (float16 on Apple GPUs) as it's read: a float32 file
    (many Civitai ControlNets are 1.4 GB float32) then never sits in memory at full size."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            out[k] = t.to(dtype) if dtype is not None and t.is_floating_point() else t
    return out


def _xl_state(path, layout, layers_per_block, dtype=None):
    """A ControlNet file's weights (SD 1.5 or SDXL), in diffusers' layout."""
    sd = _read_half(path, dtype)
    if layout == "diffusers":
        return sd
    from diffusers.loaders.single_file_utils import convert_controlnet_checkpoint
    sd = {(k[len("control_model."):] if k.startswith("control_model.") else k): v for k, v in sd.items()}
    out = convert_controlnet_checkpoint(sd, {"layers_per_block": layers_per_block})
    for a, b in (("label_emb.0.0", "add_embedding.linear_1"), ("label_emb.0.2", "add_embedding.linear_2")):
        for part in ("weight", "bias"):
            if f"{a}.{part}" in sd:
                out[f"{b}.{part}"] = sd[f"{a}.{part}"]
    return {k: v for k, v in out.items() if v is not None}


def xl_config(state, unet_config):
    """The ControlNet's shape (SD 1.5 or SDXL): the UNet's own, with as many transformer layers per block as the file has
    (diffusers' full, mid and small SDXL ControlNets differ only there) and its own conditioning encoder."""
    import re
    n_down = len(unet_config["block_out_channels"])
    depth = [0] * n_down
    for k in state:
        m = re.match(r"down_blocks\.(\d+)\.attentions\.\d+\.transformer_blocks\.(\d+)\.", k)
        if m and int(m.group(1)) < n_down:
            depth[int(m.group(1))] = max(depth[int(m.group(1))], int(m.group(2)) + 1)
    mid = 0
    for k in state:
        m = re.match(r"mid_block\.attentions\.0\.transformer_blocks\.(\d+)\.", k)
        if m:
            mid = max(mid, int(m.group(1)) + 1)
    if mid and depth[-1] and mid != depth[-1]:
        raise ValueError("its middle and last blocks have different depths, which this app can't build yet")
    # SD 1.5: the last down block has no attention but the middle block does, so only the middle block's depth
    # goes in the last slot (diffusers reads the middle block's depth from there); the down block stays plain
    down_types = tuple("CrossAttnDownBlock2D" if d else "DownBlock2D" for d in depth)
    layers = [max(1, d) for d in depth]
    if mid and not depth[-1]:
        layers[-1] = mid
    # conditioning encoder: conv_in, then pairs of (same, downsample) convs; its channel list is read from the file
    cond = []
    if "controlnet_cond_embedding.conv_in.weight" in state:
        cond.append(int(state["controlnet_cond_embedding.conv_in.weight"].shape[0]))
        i = 1
        while f"controlnet_cond_embedding.blocks.{i}.weight" in state:
            cond.append(int(state[f"controlnet_cond_embedding.blocks.{i}.weight"].shape[0]))
            i += 2
    cin = state.get("controlnet_cond_embedding.conv_in.weight")
    keep = ("in_channels", "flip_sin_to_cos", "freq_shift", "only_cross_attention", "block_out_channels",
            "layers_per_block", "downsample_padding", "mid_block_scale_factor", "act_fn", "norm_num_groups", "norm_eps",
            "cross_attention_dim", "attention_head_dim", "num_attention_heads", "use_linear_projection", "class_embed_type",
            "num_class_embeds", "upcast_attention", "resnet_time_scale_shift", "projection_class_embeddings_input_dim",
            "encoder_hid_dim", "encoder_hid_dim_type", "addition_embed_type", "addition_time_embed_dim")
    cfg = {k: unet_config[k] for k in keep if k in unet_config}
    cfg.update(down_block_types=down_types, transformer_layers_per_block=tuple(layers),
               mid_block_type="UNetMidBlock2DCrossAttn" if mid else "UNetMidBlock2D")
    if cond:
        cfg["conditioning_embedding_out_channels"] = tuple(cond)
    if cin is not None:
        cfg["conditioning_channels"] = int(cin.shape[1])
    return cfg


def _load_user_xl(kind, name, path, layout, unet, dtype, emit, family="sdxl"):
    """Your own ControlNet (SD 1.5 or SDXL): built at the file's own size (full / mid / small) and filled with its
    weights. Nothing is downloaded. The model is made empty and the file is read straight into half precision, so a
    big ControlNet never needs two or three copies of itself in memory while loading (on an 8 GB Mac that pushed
    the next pictures into swap: 30–60 s a step instead of a few)."""
    import gc
    import diffusers
    import torch
    fam = "SDXL" if family == "sdxl" else "SD 1.5"
    emit({"type": "status", "msg": f"Loading your {fam} ControlNet {name}…"})
    state = _xl_state(path, layout, int(unet.config.layers_per_block), dtype)
    cfg = xl_config(state, dict(unet.config))
    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            cn = diffusers.ControlNetModel(**cfg)
        empty = True
    except ImportError:
        cn = diffusers.ControlNetModel(**cfg)
        empty = False
    want = set(cn.state_dict().keys())
    missing = sorted(want - set(state))
    if missing:
        print(f"[controlnet] {name}: {len(missing)} missing, e.g. {missing[:6]}", flush=True)
        if len(missing) > 5:
            raise ValueError(f"“{name}” doesn't match an {fam} ControlNet's layout ({len(missing)} parts missing)")
        # a few parts missing (some older SD 1.5 files): start from the UNet's own weights for those, as before
        del cn
        gc.collect()
        cn = diffusers.ControlNetModel.from_unet(unet).to(dtype=dtype)
        empty = False
    state = {k: v.to(dtype) for k, v in state.items() if k in want}
    try:
        cn.load_state_dict(state, strict=not missing, assign=empty)
    except RuntimeError as e:
        print(f"[controlnet] {name}: {e}", flush=True)
        raise ValueError(f"“{name}” has parts of a different size than this app expected, so it can't be used")
    if any(not torch.isfinite(p).all() for p in cn.parameters()):
        raise ValueError(f"“{name}” has broken weights")
    print(f"[controlnet] {fam} {name}: {len(state)} weights, layers {cfg['transformer_layers_per_block']}, "
          f"mid {cfg['mid_block_type']}", flush=True)
    LAST_SOURCE[kind] = "user"
    del state
    gc.collect()
    return cn.to(dtype=dtype)


def _dbee_items():
    try:
        import diffusionbee
        return diffusionbee.items(force=True)
    except Exception:
        return []


def _dbee_file(hint, found=None):
    try:
        for i in (found if found is not None else _dbee_items()):
            if i["kind"] == "controlnet" and hint in Path(i["file"]).name.lower() and "inpaint" not in i["file"].lower():
                return i["file"]
    except Exception:
        pass
    return None


def is_cached(kind):
    if _dbee_file(TYPES[kind][2]):
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(TYPES[kind][1], local_files_only=True, allow_patterns=["*.json", "*.safetensors"])
        return True
    except Exception:
        return False


def load(kind, unet, dtype, emit=lambda m: None):
    """Return a diffusers ControlNetModel for this kind (SD 1.5)."""
    import diffusers
    if str(kind).startswith("user:"):
        return _load_user(kind, unet, dtype, emit)
    label, repo, hint, _ = TYPES[kind]
    dbee = None if kind in SKIP_DBEE else _dbee_file(hint)
    if dbee:
        try:
            import tdict_reader
            from diffusers.loaders.single_file_utils import convert_controlnet_checkpoint
            emit({"type": "status", "msg": f"Loading ControlNet {kind} from DiffusionBee…"})
            td = tdict_reader.TDict(dbee)
            ckpt = {k: td.tensor(k) for k in td.keys
                    if "_split_" not in k and not k.endswith(".bias_by_weight")
                    and not k.startswith(("zeros_", "ones_", "temb_coefficients", "causal_mask", "aux_output_conv",
                                          "alphas_cumprod"))}
            import torch
            cn = diffusers.ControlNetModel.from_unet(unet)
            state = convert_controlnet_checkpoint(ckpt, cn.config)
            missing, unexpected = cn.load_state_dict(state, strict=False)
            print(f"[controlnet] DiffusionBee {kind}: {len(state)} weights, {len(missing)} missing, "
                  f"{len(unexpected)} unexpected", flush=True)
            if len(missing) > 5:
                raise ValueError(f"{len(missing)} weights missing")
            # from_unet starts the ControlNet's output layers at zero: if they're still zero, the file's
            # weights didn't map and the ControlNet would do nothing useful (or wreck the picture)
            zero_out = all(float(p.detach().abs().sum()) == 0 for n, p in cn.named_parameters()
                           if n.startswith("controlnet_down_blocks.0") or n.startswith("controlnet_mid_block"))
            bad = any(not torch.isfinite(p).all() for p in cn.parameters())
            if zero_out or bad:
                raise ValueError("weights didn't load correctly")
            LAST_SOURCE[kind] = "diffusionbee"
            return cn.to(dtype=dtype)
        except Exception as e:
            print(f"[controlnet] DiffusionBee {kind} model didn't load ({e}); using the official one", flush=True)
    emit({"type": "status", "msg": f"Loading ControlNet {kind}…"})
    LAST_SOURCE[kind] = "official"
    return diffusers.ControlNetModel.from_pretrained(repo, torch_dtype=dtype, use_safetensors=True)


# ------------------------------------------------------------------ control maps
def make_map(kind, image, width, height, depth_estimator=None, already_map=False, pose_detector=None):
    """PIL image -> control map (PIL, RGB, width x height)."""
    import cv2
    from PIL import Image, ImageFilter, ImageOps
    img = image.convert("RGB").resize((width, height), Image.LANCZOS)
    if str(kind).startswith("user:"):  # your own ControlNets get the map their name asks for (else the picture)
        kind = guess_map(kind[len("user:"):]) or "tile"
    if already_map or kind == "tile":
        return img
    arr = np.asarray(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if kind == "canny":
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")
    if kind == "softedge":
        blur = cv2.GaussianBlur(gray, (0, 0), 1.6)
        gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        mag = (255 * np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)).astype(np.uint8)
        return Image.fromarray(mag).filter(ImageFilter.GaussianBlur(1)).convert("RGB")
    if kind == "lineart":
        # white lines on black, from a light edge-preserving pass (good for drawings and clear photos)
        smooth = cv2.bilateralFilter(gray, 7, 50, 50)
        lines = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
        return ImageOps.invert(Image.fromarray(lines)).convert("RGB")
    if kind == "scribble":
        # dark strokes on a light background -> white strokes on black, thickened a little
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return Image.fromarray(mask).convert("RGB")
    if kind == "openpose":
        if pose_detector is None:
            raise ValueError("pose model not loaded")
        import pose
        people = pose_detector.keypoints(image.convert("RGB"))
        if not people:
            raise ValueError("no people found in the picture to copy a pose from")
        return pose.draw(people, width, height, src_size=image.size)
    if kind == "depth":
        if depth_estimator is None:
            raise ValueError("depth model not loaded")
        out = depth_estimator(img)["depth"]
        return out.convert("RGB").resize((width, height))
    raise ValueError(f"Unknown control type {kind}")


def depth_size(size):
    return size if size in DEPTH_MODELS else "small"


def load_depth_estimator(device, size="small"):
    from transformers import pipeline
    return pipeline("depth-estimation", model=DEPTH_MODELS[depth_size(size)][0], device=device)


def depth_image(estimator, image, style="gray"):
    """Depth Anything V2 on a PIL image -> depth map at the image's own size (near = bright).
    style "color" maps it to a heat palette that's easier to read by eye."""
    from PIL import Image
    depth = estimator(image.convert("RGB"))["depth"].convert("L")
    if depth.size != image.size:
        depth = depth.resize(image.size, Image.BICUBIC)
    if style != "color":
        return depth.convert("RGB")
    import cv2
    colored = cv2.applyColorMap(np.asarray(depth), cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
