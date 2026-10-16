"""Pose detection for the OpenPose ControlNet: finds each person (RT-DETR), finds their body keypoints
(ViTPose), and draws the coloured OpenPose skeleton the ControlNet was trained on.

Both models come from transformers (no extra installs) and download once (~170 MB + ~360 MB).
"""
import math

import numpy as np

DETECTOR_REPO = "PekingU/rtdetr_r50vd_coco_o365"
POSE_REPO = "usyd-community/vitpose-base-simple"
DOWNLOAD_GB = 0.55

# COCO-17 keypoint order (ViTPose) -> OpenPose-18 order (adds the neck)
#   OpenPose: 0 nose 1 neck 2 r-shoulder 3 r-elbow 4 r-wrist 5 l-shoulder 6 l-elbow 7 l-wrist 8 r-hip 9 r-knee
#             10 r-ankle 11 l-hip 12 l-knee 13 l-ankle 14 r-eye 15 l-eye 16 r-ear 17 l-ear
_COCO_TO_OP = {0: 0, 6: 2, 8: 3, 10: 4, 5: 5, 7: 6, 9: 7, 12: 8, 14: 9, 16: 10, 11: 11, 13: 12, 15: 13,
               2: 14, 1: 15, 4: 16, 3: 17}
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


class PoseDetector:
    def __init__(self, device="cpu"):
        import torch
        from transformers import (RTDetrForObjectDetection, RTDetrImageProcessor, VitPoseForPoseEstimation,
                                  VitPoseImageProcessor)
        self.torch = torch
        # small models: the CPU is reliable and quick enough, and keeps the GPU free for the image model
        self.device = "cpu"
        self.det_proc = RTDetrImageProcessor.from_pretrained(DETECTOR_REPO)
        self.det = RTDetrForObjectDetection.from_pretrained(DETECTOR_REPO).eval()
        self.pose_proc = VitPoseImageProcessor.from_pretrained(POSE_REPO)
        self.pose = VitPoseForPoseEstimation.from_pretrained(POSE_REPO).eval()
        labels = {v.lower(): int(k) for k, v in (self.det.config.id2label or {}).items()}
        self.person_id = labels.get("person", 0)

    def people(self, image, threshold=0.4):
        """-> list of person boxes in COCO format [x, y, w, h]"""
        torch = self.torch
        inputs = self.det_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            out = self.det(**inputs)
        res = self.det_proc.post_process_object_detection(
            out, target_sizes=torch.tensor([(image.height, image.width)]), threshold=threshold)[0]
        boxes = []
        for box, label in zip(res["boxes"], res["labels"]):
            if int(label) == self.person_id:
                x1, y1, x2, y2 = [float(v) for v in box]
                boxes.append([x1, y1, x2 - x1, y2 - y1])
        return boxes

    def keypoints(self, image):
        """-> list of people; each = list of 18 (x, y, score) in OpenPose order (score 0 = missing)"""
        torch = self.torch
        image = image.convert("RGB")
        boxes = self.people(image)
        if not boxes:
            return []
        inputs = self.pose_proc(image, boxes=[boxes], return_tensors="pt")
        with torch.no_grad():
            out = self.pose(**inputs)
        results = self.pose_proc.post_process_pose_estimation(out, boxes=[boxes])[0]
        people = []
        for r in results:
            kps = r["keypoints"].cpu().numpy()
            scores = r["scores"].cpu().numpy().reshape(-1)
            op = [(0.0, 0.0, 0.0)] * 18
            for coco_i, op_i in _COCO_TO_OP.items():
                if coco_i < len(kps):
                    op[op_i] = (float(kps[coco_i][0]), float(kps[coco_i][1]), float(scores[coco_i]))
            rs, ls = op[2], op[5]
            if rs[2] > 0 and ls[2] > 0:  # neck = between the shoulders
                op[1] = ((rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2, min(rs[2], ls[2]))
            people.append(op)
        return people


def draw(people, width, height, src_size=None, min_score=0.3):
    """Draw OpenPose skeletons on black (PIL RGB, width x height)."""
    import cv2
    from PIL import Image
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    sx = width / src_size[0] if src_size else 1.0
    sy = height / src_size[1] if src_size else 1.0
    stick = max(2, int(round(4 * max(width, height) / 512)))
    for person in people:
        pts = [(x * sx, y * sy, s) for x, y, s in person]
        for i, (a, b) in enumerate(_LIMBS):
            (xa, ya, sa), (xb, yb, sb) = pts[a], pts[b]
            if sa < min_score or sb < min_score:
                continue
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            length = math.hypot(xa - xb, ya - yb)
            angle = math.degrees(math.atan2(ya - yb, xa - xb))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
            color = [int(c * 0.6) for c in _COLORS[i % len(_COLORS)]]
            cv2.fillConvexPoly(canvas, poly, color)
        for i, (x, y, s) in enumerate(pts):
            if s >= min_score:
                cv2.circle(canvas, (int(x), int(y)), stick + 1, _COLORS[i % len(_COLORS)], thickness=-1)
    return Image.fromarray(canvas)
