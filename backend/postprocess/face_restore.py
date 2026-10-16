"""Face restoration with RestoreFormer.

Pipeline: detect faces (RetinaFace, via facexlib) -> align each face to 512x512 ->
restore it with RestoreFormer -> blend with the original by `strength` -> paste back
using a face-parsing mask so edges are seamless.

Detection and parsing run on the CPU (small models, and more reliable than MPS);
RestoreFormer itself runs on the Apple GPU when available.
"""
import urllib.request
from pathlib import Path

import numpy as np

RESTOREFORMER_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"


def _download(url, dest: Path, emit=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    if emit:
        emit({"type": "status", "msg": f"Downloading {dest.name} (first time only)…"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


class FaceRestorer:
    def __init__(self, weights_dir: Path, device: str, emit=None):
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from .restoreformer_arch import RestoreFormer

        self.torch = torch
        self.device = torch.device(device)
        self.weights_dir = Path(weights_dir)
        ckpt = _download(RESTOREFORMER_URL, self.weights_dir / "RestoreFormer.pth", emit)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        self.model = RestoreFormer()
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        if emit:
            emit({"type": "status", "msg": "Preparing face detector (first time downloads ~150 MB)…"})
        self.helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
            save_ext="png", use_parse=True, device=torch.device("cpu"),
            model_rootpath=str(self.weights_dir / "facexlib"),
        )

    def restore(self, pil_image, strength=0.8):
        """Return (new PIL image, number of faces restored)."""
        import cv2
        from PIL import Image

        torch = self.torch
        strength = float(max(0.0, min(1.0, strength)))
        bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

        h = self.helper
        h.clean_all()
        h.read_image(bgr)
        h.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        h.align_warp_face()
        if not h.cropped_faces:
            return pil_image, 0

        for face in h.cropped_faces:
            x = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0  # BGR->RGB, CHW
            x = (x - 0.5) / 0.5
            with torch.no_grad():
                out = self.model(x.unsqueeze(0).to(self.device))[0]
            out = out.squeeze(0).float().clamp(-1, 1).cpu()
            out = ((out + 1) / 2 * 255.0).round().byte().permute(1, 2, 0).numpy()[:, :, ::-1]  # RGB->BGR
            blended = (strength * out.astype(np.float32) + (1 - strength) * face.astype(np.float32))
            h.add_restored_face(np.clip(blended, 0, 255).astype(np.uint8))

        h.get_inverse_affine(None)
        result = h.paste_faces_to_input_image()
        return Image.fromarray(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)), len(h.cropped_faces)
