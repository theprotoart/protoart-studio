"""ControlNet for the Image to image tab (SD 1.5).

A ControlNet makes the picture follow a "control map" made from your image: its edges (Canny),
depth, line art, a scribble, or the image itself (Tile, for adding detail / upscaling).

Models: your DiffusionBee ControlNets are used in place when you have them (offline); otherwise
the official ControlNet 1.1 model is downloaded once from Hugging Face.
"""
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
DOWNLOAD_GB = 1.45


def list_types():
    found = _dbee_items()  # scan DiffusionBee's folders once, not once per type
    out = []
    for k, (label, repo, hint, what) in TYPES.items():
        out.append({"id": k, "label": label, "map": what, "diffusionbee": bool(_dbee_file(hint, found))})
    return out


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
