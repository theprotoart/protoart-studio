"""Face swap (Image to image → 🔁 Face swap): put the face from one picture onto a person in another.

How it works (the same method as FaceFusion / roop, written for this app):
  1. RetinaFace (facexlib, already used by Restore faces) finds the faces and their 5 landmarks.
  2. ArcFace w600k_r50 turns the source face into an identity vector.
  3. InsightFace's inswapper_128 redraws each chosen target face (aligned to 128 px, or 256 px with
     "pixel boost", processed as 2 × 2 tiles of 128) with that identity.
  4. The new face is warped back and blended in with a soft mask; Restore faces (CodeFormer) can tidy it.

Models (downloaded once from Hugging Face, ~460 MB): inswapper_128_fp16.onnx and arcface_w600k_r50.onnx.
inswapper_128 is released by InsightFace for non-commercial research use only, so results are labelled that way.
Only swap faces you have permission to use.
"""
import numpy as np

REPO = "facefusion/models-3.0.0"
SWAPPER = "inswapper_128_fp16.onnx"
ARCFACE = "arcface_w600k_r50.onnx"
DOWNLOAD_MB = 455
LICENCE = "inswapper_128 (InsightFace): non-commercial research use only"

# ArcFace's five-point face template at 112 px (left eye, right eye, nose, left / right mouth corner)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def template(size):
    """InsightFace's estimate_norm: the template scaled to `size` (128 / 256 get a small sideways shift)."""
    if size % 112 == 0:
        ratio, dx = size / 112.0, 0.0
    else:
        ratio = size / 128.0
        dx = 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += dx
    return dst


def align(bgr, landmarks, size):
    """Warp the face to the template. -> (aligned BGR image, 2×3 matrix)"""
    import cv2
    M, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), template(size), method=cv2.LMEDS)
    if M is None:
        raise ValueError("couldn't line up that face")
    return cv2.warpAffine(bgr, M, (size, size), borderValue=0.0), M


def is_cached():
    from huggingface_hub import try_to_load_from_cache
    return all(isinstance(try_to_load_from_cache(REPO, f), str) for f in (SWAPPER, ARCFACE))


class FaceSwapper:
    def __init__(self, face_net, emit=None):
        """face_net: facexlib's RetinaFace (detailer.FaceFinder(...).net)."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        self.emit = emit or (lambda m: None)
        self.face_net = face_net
        if not is_cached():
            self.emit({"type": "status", "msg": f"Downloading the face swap models (first time only, ~{DOWNLOAD_MB} MB)…"})
        swapper_path = hf_hub_download(REPO, SWAPPER)
        arcface_path = hf_hub_download(REPO, ARCFACE)
        prov = ["CPUExecutionProvider"]  # CoreML gives no gain for these small models and can fail to compile
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.swapper = ort.InferenceSession(swapper_path, sess_options=opts, providers=prov)
        self.arcface = ort.InferenceSession(arcface_path, sess_options=opts, providers=prov)
        self.emap = self._emap(swapper_path)
        self.in_names = [i.name for i in self.swapper.get_inputs()]
        self.in_types = {i.name: i.type for i in self.swapper.get_inputs()}

    @staticmethod
    def _emap(path):
        """The swapper's identity mapping matrix (the model file's last stored weight)."""
        import onnx
        from onnx import numpy_helper
        model = onnx.load(path)
        return numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)

    # ---------------------------------------------------------------- faces
    def faces(self, image, min_score=0.6):
        """-> list of {box, score, landmarks (5×2)} biggest first."""
        import cv2
        import torch
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            dets = self.face_net.detect_faces(bgr, conf_threshold=min_score)
        out = []
        for d in dets if dets is not None else []:
            d = [float(v) for v in d]
            box = d[:4]
            if min(box[2] - box[0], box[3] - box[1]) < 16:
                continue
            out.append({"box": box, "score": d[4], "landmarks": np.array(d[5:15], np.float32).reshape(5, 2)})
        out.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), reverse=True)
        return out

    def identity(self, image, face):
        """ArcFace identity of one face, mapped for the swapper (normalised)."""
        import cv2
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        crop, _ = align(bgr, face["landmarks"], 112)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.arcface.run(None, {self.arcface.get_inputs()[0].name: blob})[0].reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        latent = emb.reshape(1, -1) @ self.emap
        return (latent / (np.linalg.norm(latent) + 1e-8)).astype(np.float32)

    # ---------------------------------------------------------------- swap
    def _run(self, crop128_bgr, latent):
        import cv2
        blob = cv2.dnn.blobFromImage(crop128_bgr, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        feeds = {}
        for name in self.in_names:
            arr = blob if name == "target" else latent
            if "float16" in self.in_types[name]:
                arr = arr.astype(np.float16)
            feeds[name] = arr
        out = self.swapper.run(None, feeds)[0][0].astype(np.float32)  # RGB, 0..1
        out = np.clip(out.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return out[:, :, ::-1]  # → BGR

    def swap_face(self, bgr, face, latent, boost=1):
        """Swap one face in a BGR image (in place copy). boost 1 = 128 px, 2 = 256 px (4 tiles)."""
        import cv2
        size = 128 * boost
        crop, M = align(bgr, face["landmarks"], size)
        if boost == 1:
            fake = self._run(crop, latent)
        else:
            fake = np.zeros_like(crop)
            for y in range(boost):  # the swapper works at 128 px: do the face as tiles, every tile the same identity
                for x in range(boost):
                    tile = crop[y::boost, x::boost]
                    fake[y::boost, x::boost] = self._run(np.ascontiguousarray(tile), latent)
        h, w = bgr.shape[:2]
        IM = cv2.invertAffineTransform(M)
        warped = cv2.warpAffine(fake, IM, (w, h), borderValue=0.0)
        # soft mask: the aligned square, shrunk and feathered so the edge never shows
        mask = np.full((size, size), 255, np.float32)
        pad = int(size * 0.04)
        mask[:pad, :] = mask[-pad:, :] = 0
        mask[:, :pad] = mask[:, -pad:] = 0
        mask = cv2.warpAffine(mask, IM, (w, h), borderValue=0.0)
        ys, xs = np.where(mask > 20)
        if len(ys) == 0:
            return bgr
        msize = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
        k = max(msize // 10, 10)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        k = max(msize // 20, 5)
        mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)[:, :, None] / 255.0
        return (mask * warped.astype(np.float32) + (1 - mask) * bgr.astype(np.float32)).clip(0, 255).astype(np.uint8)

    def swap(self, source_image, target_image, which="all", source_index=0, boost=1):
        """Swap the source picture's face onto the target picture.
        which: "all" faces, "biggest", or an index (0 = biggest). -> (PIL image, number of faces swapped)"""
        import cv2
        from PIL import Image
        src_faces = self.faces(source_image)
        if not src_faces:
            raise ValueError("no face found in the face picture (use a clear, front-facing photo)")
        src = src_faces[min(int(source_index or 0), len(src_faces) - 1)]
        latent = self.identity(source_image, src)
        tgt_faces = self.faces(target_image)
        if not tgt_faces:
            raise ValueError("no face found in the picture to change")
        if which == "biggest":
            chosen = tgt_faces[:1]
        elif isinstance(which, int) or (isinstance(which, str) and which.isdigit()):
            i = int(which)
            chosen = [tgt_faces[min(i, len(tgt_faces) - 1)]]
        else:
            chosen = tgt_faces
        bgr = cv2.cvtColor(np.asarray(target_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        for n, f in enumerate(chosen):
            self.emit({"type": "status", "msg": f"Swapping face {n + 1} of {len(chosen)}…"})
            self.emit({"type": "progress", "image": 1, "images": 1, "step": n, "steps": len(chosen)})
            bgr = self.swap_face(bgr, f, latent, boost=max(1, min(2, int(boost or 1))))
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), len(chosen)
