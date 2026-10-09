from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib
import numpy as np
from ..candidate_filter import compute_aspect_ratio, diameter_validity, shape_aspect_ratio_score
from ..contracts import BallCandidate2D


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class SoccerNetV3DYOLOProvider:
    name = "soccernet-v3d-yolo"

    def __init__(
        self,
        weights: str | Path,
        *,
        conf_floor: float = 0.05,
        top_k: int = 10,
        imgsz: int = 1920,
        device: str = "cpu",
        iou: float = 0.5,
        max_aspect_ratio: float = 1.60,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError('Install YOLO support with: pip install -e ".[yolo]"') from exc
        self.weights = str(Path(weights).expanduser().resolve())
        if not Path(self.weights).is_file():
            raise FileNotFoundError(self.weights)
        self.conf_floor = float(conf_floor)
        self.top_k = int(top_k)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.iou = float(iou)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.model = YOLO(self.weights)
        self.weights_sha256 = sha256_file(self.weights)

    def detect(self, image_bgr: np.ndarray, frame_index: int) -> list[BallCandidate2D]:
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_floor,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=max(self.top_k * 4, self.top_k),
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(conf))

        candidates = []
        for idx in range(len(conf)):
            x1, y1, x2, y2 = [float(x) for x in xyxy[idx]]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 2.0 or h < 2.0:
                continue
            ar = compute_aspect_ratio([x1, y1, x2, y2])
            if ar > self.max_aspect_ratio:
                continue
            d = 0.5 * (w + h)
            if d < 2.5 or d > 60.0:
                continue
            shape_score = shape_aspect_ratio_score([x1, y1, x2, y2], max_aspect_ratio=self.max_aspect_ratio)
            # Prioritize candidates that are spherical over elongated net mesh cords
            adjusted_score = float(conf[idx]) * (0.6 + 0.4 * shape_score)
            candidates.append({
                "idx": idx,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "conf": float(conf[idx]),
                "adjusted_score": adjusted_score,
                "class_id": int(cls[idx]),
                "diameter": d,
                "ar": ar,
                "shape_score": shape_score,
            })

        # Sort by adjusted score to keep top_k candidates
        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        selected = candidates[:self.top_k]

        out = []
        for rank, c in enumerate(selected, start=1):
            out.append(
                BallCandidate2D(
                    int(frame_index),
                    f"ball_{frame_index:08d}_{rank:02d}",
                    c["bbox"],
                    c["center"],
                    c["conf"],
                    "yolo-sn-v3d",
                    c["diameter"],
                    metadata={
                        "class_id": c["class_id"],
                        "rank_by_detector_score": rank,
                        "aspect_ratio": c["ar"],
                        "shape_score": c["shape_score"],
                    },
                )
            )
        return out

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": self.weights,
            "weights_sha256": self.weights_sha256,
            "conf_floor": self.conf_floor,
            "top_k": self.top_k,
            "imgsz": self.imgsz,
            "device": self.device,
            "nms_iou": self.iou,
            "max_aspect_ratio": self.max_aspect_ratio,
        }
