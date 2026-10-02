from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import cv2
import numpy as np


def compute_crop_geometry(
    bbox: Sequence[float],
    *,
    input_width: int,
    input_height: int,
    bbox_padding: float = 1.25,
    crop_scale: float = 1.0,
) -> dict:
    """Return the exact crop geometry shared by inference and crop QA."""
    x1, y1, x2, y2 = map(float, bbox)
    center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.asarray([max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32) * float(bbox_padding) * float(crop_scale)
    aspect = float(input_width) / float(input_height)
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    src_dir = np.asarray([0.0, -0.5 * float(scale[0])], dtype=np.float32)
    dst_dir = np.asarray([0.0, -0.5 * float(input_width)], dtype=np.float32)
    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + src_dir
    source_direction = source[0] - source[1]
    source[2] = source[1] + np.asarray([-source_direction[1], source_direction[0]], dtype=np.float32)
    destination[0] = (0.5 * float(input_width), 0.5 * float(input_height))
    destination[1] = destination[0] + dst_dir
    destination_direction = destination[0] - destination[1]
    destination[2] = destination[1] + np.asarray([-destination_direction[1], destination_direction[0]], dtype=np.float32)
    affine = cv2.getAffineTransform(source, destination)
    rect = [float(center[0] - scale[0] * 0.5), float(center[1] - scale[1] * 0.5), float(center[0] + scale[0] * 0.5), float(center[1] + scale[1] * 0.5)]
    return {
        "crop_center_xy": [float(x) for x in center],
        "crop_scale_xy": [float(x) for x in scale],
        "crop_rect_xyxy": rect,
        "affine_matrix_source_to_model": [[float(x) for x in row] for row in affine],
    }

@dataclass
class PoseResult:
    keypoints_xy: np.ndarray
    scores: np.ndarray
    source_bbox_xyxy: Tuple[float, float, float, float]
    crop_scale: float
    inference_diagnostics: dict

class RTMWOpenCVDNN:
    """Minimal self-contained RTMW SimCC ONNX runner.

    Input defaults match the RTMW-L 384x288 graph used in Stage 2:
    width=288, height=384. Scores are raw SimCC maxima, not probabilities.
    """
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_width: int = 288,
        input_height: int = 384,
        bbox_padding: float = 1.25,
        simcc_split_ratio: float = 2.0,
        device: str = "cpu",
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.input_size = np.asarray([self.input_width, self.input_height], dtype=np.float32)
        self.bbox_padding = float(bbox_padding)
        self.simcc_split_ratio = float(simcc_split_ratio)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        if device == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            raise ValueError("device must be cpu or cuda")
        self.device = device
        self.output_names = list(self.net.getUnconnectedOutLayersNames())

    @staticmethod
    def _center_scale(bbox: Sequence[float], padding: float) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = map(float, bbox)
        center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        scale = np.asarray([max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32) * padding
        return center, scale

    @staticmethod
    def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        return b + np.asarray([-d[1], d[0]], dtype=np.float32)

    @classmethod
    def _warp_matrix(cls, center: np.ndarray, scale: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        dst_width, dst_height = map(float, output_size)
        src_dir = np.asarray([0.0, -0.5 * float(scale[0])], dtype=np.float32)
        dst_dir = np.asarray([0.0, -0.5 * dst_width], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0] = center
        src[1] = center + src_dir
        src[2] = cls._third_point(src[0], src[1])
        dst[0] = (0.5 * dst_width, 0.5 * dst_height)
        dst[1] = dst[0] + dst_dir
        dst[2] = cls._third_point(dst[0], dst[1])
        return cv2.getAffineTransform(src, dst)

    def _preprocess(self, image: np.ndarray, bbox: Sequence[float], crop_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        geometry = compute_crop_geometry(
            bbox,
            input_width=self.input_width,
            input_height=self.input_height,
            bbox_padding=self.bbox_padding,
            crop_scale=crop_scale,
        )
        center = np.asarray(geometry["crop_center_xy"], dtype=np.float32)
        scale = np.asarray(geometry["crop_scale_xy"], dtype=np.float32)
        M = np.asarray(geometry["affine_matrix_source_to_model"], dtype=np.float32)
        warped = cv2.warpAffine(image, M, (self.input_width, self.input_height), flags=cv2.INTER_LINEAR).astype(np.float32)
        normalized = (warped - self.mean) / self.std
        blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        return blob, center, scale

    def _identify_axes(self, outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        arrays = []
        for arr in outputs:
            x = np.asarray(arr)
            while x.ndim > 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2:
                x = x[None]
            arrays.append(x)
        ex = int(round(self.input_width * self.simcc_split_ratio))
        ey = int(round(self.input_height * self.simcc_split_ratio))
        xc = [x for x in arrays if x.shape[-1] == ex]
        yc = [x for x in arrays if x.shape[-1] == ey]
        if xc and yc:
            return xc[0], yc[0]
        arrays = sorted(arrays, key=lambda x: x.shape[-1])
        if len(arrays) < 2:
            raise RuntimeError("RTMW ONNX did not return two SimCC tensors")
        return arrays[0], arrays[-1]

    def infer_one(self, image: np.ndarray, bbox_xyxy: Sequence[float], *, crop_scale: float = 1.0) -> PoseResult:
        blob, center, scale = self._preprocess(image, bbox_xyxy, crop_scale)
        self.net.setInput(blob)
        raw = self.net.forward(self.output_names) if self.output_names else self.net.forward()
        outputs = [raw] if isinstance(raw, np.ndarray) else [np.asarray(x) for x in raw]
        sx, sy = self._identify_axes(outputs)
        if sx.shape[:2] != sy.shape[:2]:
            raise RuntimeError(f"Mismatched RTMW tensors: {sx.shape} vs {sy.shape}")
        b, k, _ = sx.shape
        xflat, yflat = sx.reshape(b * k, -1), sy.reshape(b * k, -1)
        xloc, yloc = np.argmax(xflat, axis=1), np.argmax(yflat, axis=1)
        scores = (0.5 * (np.max(xflat, axis=1) + np.max(yflat, axis=1))).reshape(b, k).astype(np.float32)
        loc = np.stack([xloc, yloc], axis=-1).astype(np.float32).reshape(b, k, 2)
        loc[scores <= 0] = -1
        xy = loc / self.simcc_split_ratio
        xy = xy / self.input_size * scale
        xy = xy + center - scale / 2.0
        if k != 133:
            raise RuntimeError(f"Expected 133 RTMW keypoints, got {k}")
        finite_xy = np.isfinite(xy[0]).all(axis=1)
        finite_scores = np.isfinite(scores[0])
        positive_scores = scores[0][finite_scores & (scores[0] > 0.0)]
        diagnostics = {
            "backend": "opencv_dnn",
            "device": self.device,
            "input_size_wh": [self.input_width, self.input_height],
            "bbox_padding": self.bbox_padding,
            "crop_scale": float(crop_scale),
            "source_bbox_xyxy": [float(x) for x in bbox_xyxy],
            "crop_center_xy": [float(x) for x in center],
            "crop_scale_xy": [float(x) for x in scale],
            "crop_geometry": compute_crop_geometry(
                bbox_xyxy,
                input_width=self.input_width,
                input_height=self.input_height,
                bbox_padding=self.bbox_padding,
                crop_scale=crop_scale,
            ),
            "image_size_hwc": [int(x) for x in image.shape],
            "input_normalization": {"mean_bgr": [float(x) for x in self.mean.reshape(-1)], "std_bgr": [float(x) for x in self.std.reshape(-1)]},
            "simcc_split_ratio": self.simcc_split_ratio,
            "output_shapes": [list(np.asarray(x).shape) for x in outputs],
            "keypoint_count": int(k),
            "finite_keypoint_count": int(np.sum(finite_xy)),
            "positive_score_count": int(positive_scores.size),
            "raw_score_min": None if positive_scores.size == 0 else float(np.min(positive_scores)),
            "raw_score_median": None if positive_scores.size == 0 else float(np.median(positive_scores)),
            "raw_score_max": None if positive_scores.size == 0 else float(np.max(positive_scores)),
        }
        return PoseResult(
            xy[0].astype(np.float32),
            scores[0],
            tuple(map(float, bbox_xyxy)),
            float(crop_scale),
            diagnostics,
        )
