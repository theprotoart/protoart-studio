"""Face restoration with RestoreFormer (default) or CodeFormer.

Pipeline: detect faces (RetinaFace, via facexlib) -> align each face to 512x512 ->
restore it with the chosen model -> blend with the original by `strength` -> paste back
using a face-parsing mask so edges are seamless.

Detection and parsing run on the CPU (small models, and more reliable than MPS);
the restoration model itself runs on the Apple GPU when available.

Both model files are official release downloads, checked against pinned SHA-256 hashes before
they are loaded (with torch's weights_only loader, so they can't run code).
"""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

MODELS = {
    # TencentARC/GFPGAN v1.3.4 release (Apache-2.0 code; RestoreFormer by Wang et al., CVPR 2022)
    "restoreformer": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "file": "RestoreFormer.pth",
        "sha256": "07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0",
        "label": "RestoreFormer",
        "mb": 278,
    },
    # sczhou/CodeFormer v0.1.0 release (S-Lab License 1.0: non-commercial use only)
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "file": "codeformer.pth",
        "sha256": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        "label": "CodeFormer",
        "mb": 360,
    },
}

# Kept for anything that imported the old name.
RESTOREFORMER_URL = MODELS["restoreformer"]["url"]


def model_key(name) -> str:
    return "restoreformer" if str(name or "").lower() == "restoreformer" else "codeformer"  # CodeFormer is the default


def model_label(name) -> str:
    return MODELS[model_key(name)]["label"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest: Path, emit=None, sha256=None, label=None, mb=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256 is None or _sha256(dest) == sha256:
            return dest
        dest.unlink()  # damaged or tampered: fetch a fresh copy
    if emit:
        size = f" (~{mb} MB)" if mb else ""
        emit({"type": "status", "msg": f"Downloading {label or dest.name}{size}, first time only…"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256 is not None and _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"The downloaded {label or dest.name} failed its security check (SHA-256 mismatch) "
                           "and was deleted. Check your connection and try again.")
    tmp.rename(dest)
    return dest


class FaceRestorer:
    def __init__(self, weights_dir: Path, device: str, emit=None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.weights_dir = Path(weights_dir)
        self.emit = emit
        self.models = {}   # one restoration model kept loaded at a time
        self.helper = None

    def _get_helper(self):
        if self.helper is None:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            if self.emit:
                self.emit({"type": "status", "msg": "Preparing face detector (first time downloads ~150 MB)…"})
            self.helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
                save_ext="png", use_parse=True, device=self.torch.device("cpu"),
                model_rootpath=str(self.weights_dir / "facexlib"),
            )
        return self.helper

    def _get_model(self, key: str):
        if key in self.models:
            return self.models[key]
        torch = self.torch
        info = MODELS[key]
        ckpt = _download(info["url"], self.weights_dir / info["file"], self.emit,
                         sha256=info["sha256"], label=info["label"], mb=info["mb"])
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        if key == "codeformer":
            from .codeformer_arch import CodeFormer
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                             connect_list=["32", "64", "128", "256"])
        else:
            from .restoreformer_arch import RestoreFormer
            net = RestoreFormer()
        net.load_state_dict(state, strict=True)
        del state
        net.eval().to(self.device)
        self.models.clear()  # keep memory low on 8 GB Macs
        self.models[key] = net
        return net

    def restore(self, pil_image, strength=0.8, model="restoreformer", fidelity=0.5):
        """Return (new PIL image, number of faces restored).

        model: "restoreformer" (default) or "codeformer". fidelity (CodeFormer only, 0-1): higher keeps more
        of the person's own features, lower gives a cleaner face.
        """
        import cv2
        from PIL import Image

        torch = self.torch
        key = model_key(model)
        strength = float(max(0.0, min(1.0, strength)))
        fidelity = float(max(0.0, min(1.0, fidelity)))
        bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

        h = self._get_helper()
        h.clean_all()
        h.read_image(bgr)
        h.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        h.align_warp_face()
        if not h.cropped_faces:
            return pil_image, 0

        net = self._get_model(key)
        for face in h.cropped_faces:
            x = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0  # BGR->RGB, CHW
            x = (x - 0.5) / 0.5
            with torch.no_grad():
                if key == "codeformer":
                    out = net(x.unsqueeze(0).to(self.device), w=fidelity, adain=True)[0]
                else:
                    out = net(x.unsqueeze(0).to(self.device))[0]
            out = out.squeeze(0).float().clamp(-1, 1).cpu()
            out = ((out + 1) / 2 * 255.0).round().byte().permute(1, 2, 0).numpy()[:, :, ::-1]  # RGB->BGR
            blended = (strength * out.astype(np.float32) + (1 - strength) * face.astype(np.float32))
            h.add_restored_face(np.clip(blended, 0, 255).astype(np.uint8))

        h.get_inverse_affine(None)
        result = h.paste_faces_to_input_image()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return Image.fromarray(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)), len(h.cropped_faces)
