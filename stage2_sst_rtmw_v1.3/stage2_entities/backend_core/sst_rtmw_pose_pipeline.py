#!/usr/bin/env python3
"""SST + RTMW whole-body pose pipeline for football broadcast frames.

The pipeline is intentionally modular:

1. SST/Faster R-CNN detects the ball, players, goalkeepers, referees and staff.
2. Class-aware filtering keeps person boxes and one or more ball candidates.
3. A top-down pose backend receives each expanded player box.
4. RTMW whole-body keypoints are mapped back to the original frame.
5. Legal body points for offside analysis are marked, while elbows/wrists/hands
   are explicitly excluded.
6. An annotated image and machine-readable JSON are produced.

The real pose backend uses an RTMW ONNX file through OpenCV DNN.  A clearly
watermarked bbox_fallback backend is included only to smoke-test the complete
plumbing before an ONNX model is available; it is not a learned pose estimator
and must never be used for evaluation.

Security note: the legacy SST checkpoint contains a pickled Python object.
Only load checkpoints from a trusted source.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .sst_single_frame_inference import (
    CLASS_NAMES,
    build_sst_model,
    predict_frame,
    resolve_device,
)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

# COCO-WholeBody ordering.  RTMW predicts 133 points:
# 17 body + 6 feet + 68 face + 21 left hand + 21 right hand.
COCO_BODY_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_FOOT_NAMES: Tuple[str, ...] = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

WHOLEBODY_KEYPOINT_NAMES: Tuple[str, ...] = (
    COCO_BODY_NAMES
    + COCO_FOOT_NAMES
    + tuple(f"face_{idx:02d}" for idx in range(68))
    + tuple(f"left_hand_{idx:02d}" for idx in range(21))
    + tuple(f"right_hand_{idx:02d}" for idx in range(21))
)

if len(WHOLEBODY_KEYPOINT_NAMES) != 133:
    raise RuntimeError("COCO-WholeBody keypoint name table must contain 133 entries.")

# Legal for the first implementation of offside-position geometry:
# - head/face, shoulders/torso, hips, knees, ankles and feet are included;
# - elbows, wrists and both hand groups are excluded.
# Shoulder keypoints are an approximation of the torso/arm boundary; a future
# human-part mask is still required for a strict bottom-of-armpit boundary.
LEGAL_BODY_INDICES = frozenset(
    list(range(0, 7))  # head and shoulders
    + list(range(11, 23))  # hips, legs, ankles and feet
    + list(range(23, 91))  # face landmarks
)
EXCLUDED_ARM_INDICES = frozenset([7, 8, 9, 10] + list(range(91, 133)))

# Edges kept deliberately sparse so that small broadcast players remain visible.
BODY_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 17),
    (15, 18),
    (15, 19),
    (16, 20),
    (16, 21),
    (16, 22),
)

PERSON_CLASS_IDS = frozenset({2, 3})
BALL_CLASS_ID = 1

# BGR drawing colors.  They are selected for contrast on green pitches.
COLOR_BALL = (0, 225, 255)
COLOR_PLAYER = (255, 210, 0)
COLOR_GOALKEEPER = (60, 80, 255)
COLOR_OTHER = (255, 0, 255)
COLOR_LEGAL = (0, 255, 255)
COLOR_EXCLUDED = (185, 185, 185)
COLOR_TORSO = (255, 0, 220)
COLOR_TEXT = (20, 20, 20)
COLOR_TEXT_BG = (245, 245, 245)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    detection_id: str
    label_id: int
    label: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class PoseResult:
    keypoints_xy: np.ndarray  # [K, 2], original-frame coordinates
    scores: np.ndarray  # [K]
    backend: str
    model_input_size: Tuple[int, int]
    source_bbox_xyxy: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PoseQuality:
    score: float
    level: str
    visible_core_points: int
    total_core_points: int
    visible_core_ratio: float
    mean_core_confidence: float
    bbox_height_px: float


# ---------------------------------------------------------------------------
# Geometry and filtering helpers
# ---------------------------------------------------------------------------


def clip_box(
    box: Sequence[float], image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    """Clip xyxy box to the image while preserving a positive area."""

    if len(box) != 4:
        raise ValueError(f"Expected four box coordinates, got {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = min(max(x1, 0.0), max(0.0, image_width - 1.0))
    y1 = min(max(y1, 0.0), max(0.0, image_height - 1.0))
    x2 = min(max(x2, x1 + 1.0), float(image_width))
    y2 = min(max(y2, y1 + 1.0), float(image_height))
    return x1, y1, x2, y2


def expand_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    factor: float = 1.20,
) -> Tuple[float, float, float, float]:
    """Expand an xyxy box around its center and clip it to the frame."""

    if factor < 1.0:
        raise ValueError("box expansion factor must be >= 1.0")
    x1, y1, x2, y2 = clip_box(box, image_width, image_height)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    width = max(1.0, x2 - x1) * factor
    height = max(1.0, y2 - y1) * factor
    return clip_box(
        (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0),
        image_width,
        image_height,
    )


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def greedy_nms(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    """Simple class-local greedy NMS, sorted by score descending."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be in [0, 1].")
    candidates = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: List[Detection] = []
    while candidates:
        current = candidates.pop(0)
        kept.append(current)
        candidates = [
            candidate
            for candidate in candidates
            if box_iou(current.bbox_xyxy, candidate.bbox_xyxy) <= iou_threshold
        ]
    return kept


def postprocess_sst_prediction(
    prediction: Mapping[str, torch.Tensor],
    image_width: int,
    image_height: int,
    *,
    class_thresholds: Mapping[int, float],
    person_nms_iou: float = 0.65,
    ball_nms_iou: float = 0.30,
    ball_top_k: int = 1,
    max_per_class: int = 100,
) -> List[Detection]:
    """Convert raw SST tensors into filtered, class-aware detections."""

    boxes = prediction.get("boxes", torch.empty((0, 4)))
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.long))
    scores = prediction.get("scores", torch.empty((0,)))

    grouped: MutableMapping[int, List[Detection]] = {}
    running_ids: MutableMapping[int, int] = {}
    for box_tensor, label_tensor, score_tensor in zip(boxes, labels, scores):
        label_id = int(label_tensor.item())
        score = float(score_tensor.item())
        threshold = float(class_thresholds.get(label_id, 1.0))
        if score < threshold:
            continue
        box = clip_box(box_tensor.tolist(), image_width, image_height)
        running_ids[label_id] = running_ids.get(label_id, 0) + 1
        prefix = {
            1: "ball",
            2: "player",
            3: "goalkeeper",
            4: "main_referee",
            5: "side_referee",
            6: "staff",
        }.get(label_id, f"class_{label_id}")
        grouped.setdefault(label_id, []).append(
            Detection(
                detection_id=f"{prefix}_{running_ids[label_id]:03d}",
                label_id=label_id,
                label=CLASS_NAMES.get(label_id, f"Class {label_id}"),
                score=score,
                bbox_xyxy=box,
            )
        )

    kept: List[Detection] = []
    for label_id, items in grouped.items():
        nms_threshold = ball_nms_iou if label_id == BALL_CLASS_ID else person_nms_iou
        class_kept = greedy_nms(items, nms_threshold)
        if label_id == BALL_CLASS_ID:
            class_kept = class_kept[: max(0, ball_top_k)]
        else:
            class_kept = class_kept[:max_per_class]
        kept.extend(class_kept)

    # Stable visualization/order: people first, then ball, then other classes;
    # within each class use score descending.
    order = {2: 0, 3: 1, 1: 2, 4: 3, 5: 4, 6: 5}
    kept.sort(key=lambda item: (order.get(item.label_id, 99), -item.score))

    # Re-number IDs after NMS so JSON is compact and deterministic.
    counters: MutableMapping[int, int] = {}
    normalized: List[Detection] = []
    for item in kept:
        counters[item.label_id] = counters.get(item.label_id, 0) + 1
        prefix = item.detection_id.rsplit("_", 1)[0]
        normalized.append(
            Detection(
                detection_id=f"{prefix}_{counters[item.label_id]:03d}",
                label_id=item.label_id,
                label=item.label,
                score=item.score,
                bbox_xyxy=item.bbox_xyxy,
            )
        )
    return normalized


# ---------------------------------------------------------------------------
# Pose backends
# ---------------------------------------------------------------------------


class PoseEstimator(ABC):
    name: str
    is_learned_model: bool

    @abstractmethod
    def infer(
        self, image_bgr: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]
    ) -> List[PoseResult]:
        """Infer one pose per top-down person box."""


class RTMWOpenCVDNNPose(PoseEstimator):
    """RTMW/RTMPose SimCC inference with only OpenCV and NumPy.

    This implementation follows the public rtmlib preprocessing/postprocessing:
    bbox padding 1.25, aspect-ratio-preserving affine warp, ImageNet-style
    normalization, SimCC argmax decoding, and mapping back to the source box.
    """

    name = "rtmw_opencv_dnn"
    is_learned_model = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_width: int = 192,
        input_height: int = 256,
        simcc_split_ratio: float = 2.0,
        bbox_padding: float = 1.25,
        device: str = "cpu",
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
    ) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"RTMW ONNX model not found: {model_path}. "
                "Run download_rtmw_model.py or provide --rtmw-model."
            )
        if input_width <= 0 or input_height <= 0:
            raise ValueError("RTMW input dimensions must be positive.")
        if simcc_split_ratio <= 0:
            raise ValueError("simcc_split_ratio must be positive.")
        if bbox_padding <= 0:
            raise ValueError("bbox_padding must be positive.")
        self.model_path = model_path
        self.input_size = np.array([float(input_width), float(input_height)], dtype=np.float32)
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.simcc_split_ratio = float(simcc_split_ratio)
        self.bbox_padding = float(bbox_padding)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        if device == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == "cuda":
            # Requires an OpenCV build compiled with CUDA.
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            raise ValueError("RTMW device must be 'cpu' or 'cuda'.")
        self.device = device
        self.output_names = list(self.net.getUnconnectedOutLayersNames())

    @staticmethod
    def _bbox_xyxy_to_center_scale(
        bbox: Sequence[float], padding: float = 1.25
    ) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        scale = np.asarray([x2 - x1, y2 - y1], dtype=np.float32) * float(padding)
        return center, scale

    @staticmethod
    def _rotate_point(point: np.ndarray, angle_radians: float) -> np.ndarray:
        sine, cosine = np.sin(angle_radians), np.cos(angle_radians)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        return rotation @ point

    @staticmethod
    def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = a - b
        return b + np.asarray([-direction[1], direction[0]], dtype=np.float32)

    @classmethod
    def _warp_matrix(
        cls,
        center: np.ndarray,
        scale: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        src_width = float(scale[0])
        dst_width, dst_height = float(output_size[0]), float(output_size[1])
        src_direction = cls._rotate_point(
            np.asarray([0.0, -0.5 * src_width], dtype=np.float32), 0.0
        )
        dst_direction = np.asarray([0.0, -0.5 * dst_width], dtype=np.float32)

        source = np.zeros((3, 2), dtype=np.float32)
        destination = np.zeros((3, 2), dtype=np.float32)
        source[0] = center
        source[1] = center + src_direction
        source[2] = cls._third_point(source[0], source[1])
        destination[0] = (0.5 * dst_width, 0.5 * dst_height)
        destination[1] = destination[0] + dst_direction
        destination[2] = cls._third_point(destination[0], destination[1])
        return cv2.getAffineTransform(source, destination)

    def preprocess(
        self, image_bgr: np.ndarray, bbox_xyxy: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        center, scale = self._bbox_xyxy_to_center_scale(
            bbox_xyxy, padding=self.bbox_padding
        )
        aspect_ratio = self.input_width / self.input_height
        box_width, box_height = float(scale[0]), float(scale[1])
        if box_width > box_height * aspect_ratio:
            scale[1] = box_width / aspect_ratio
        else:
            scale[0] = box_height * aspect_ratio

        affine = self._warp_matrix(center, scale, (self.input_width, self.input_height))
        warped = cv2.warpAffine(
            image_bgr,
            affine,
            (self.input_width, self.input_height),
            flags=cv2.INTER_LINEAR,
        ).astype(np.float32)
        normalized = (warped - self.mean) / self.std
        blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        return blob, center, scale

    def _forward(self, blob: np.ndarray) -> List[np.ndarray]:
        self.net.setInput(blob)
        if self.output_names:
            raw = self.net.forward(self.output_names)
        else:
            raw = self.net.forward()
        if isinstance(raw, np.ndarray):
            return [raw]
        return [np.asarray(item) for item in raw]

    def _identify_simcc_axes(self, outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Find x/y SimCC outputs even when ONNX output order differs."""

        arrays = [np.asarray(output) for output in outputs]
        expected_x = int(round(self.input_width * self.simcc_split_ratio))
        expected_y = int(round(self.input_height * self.simcc_split_ratio))

        def normalized_shape(array: np.ndarray) -> np.ndarray:
            result = array
            while result.ndim > 3 and result.shape[0] == 1:
                result = result[0]
            if result.ndim == 2:
                result = result[None]
            return result

        arrays = [normalized_shape(array) for array in arrays]
        if len(arrays) < 2:
            raise RuntimeError(
                f"RTMW model returned {len(arrays)} output(s); expected two SimCC tensors."
            )

        x_candidates = [array for array in arrays if array.shape[-1] == expected_x]
        y_candidates = [array for array in arrays if array.shape[-1] == expected_y]
        if x_candidates and y_candidates:
            return x_candidates[0], y_candidates[0]

        # Fallback: x should have the shorter last axis for portrait inputs.
        arrays = sorted(arrays, key=lambda array: int(array.shape[-1]))
        simcc_x, simcc_y = arrays[0], arrays[-1]
        if simcc_x.ndim != 3 or simcc_y.ndim != 3:
            raise RuntimeError(
                "Unexpected RTMW output shapes: "
                + ", ".join(str(tuple(array.shape)) for array in arrays)
            )
        return simcc_x, simcc_y

    def decode(
        self,
        outputs: Sequence[np.ndarray],
        center: np.ndarray,
        scale: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        simcc_x, simcc_y = self._identify_simcc_axes(outputs)
        if simcc_x.shape[0] != simcc_y.shape[0] or simcc_x.shape[1] != simcc_y.shape[1]:
            raise RuntimeError(
                f"Mismatched SimCC shapes: x={simcc_x.shape}, y={simcc_y.shape}"
            )
        batch, keypoints, _ = simcc_x.shape
        flat_x = simcc_x.reshape(batch * keypoints, -1)
        flat_y = simcc_y.reshape(batch * keypoints, -1)
        x_locations = np.argmax(flat_x, axis=1)
        y_locations = np.argmax(flat_y, axis=1)
        max_x = np.amax(flat_x, axis=1)
        max_y = np.amax(flat_y, axis=1)
        scores = (0.5 * (max_x + max_y)).reshape(batch, keypoints).astype(np.float32)
        locations = np.stack((x_locations, y_locations), axis=-1).astype(np.float32)
        locations = locations.reshape(batch, keypoints, 2)
        locations[scores <= 0.0] = -1.0

        keypoints_xy = locations / self.simcc_split_ratio
        keypoints_xy = keypoints_xy / self.input_size * scale
        keypoints_xy = keypoints_xy + center - scale / 2.0
        return keypoints_xy.astype(np.float32), scores

    def infer(
        self, image_bgr: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]
    ) -> List[PoseResult]:
        results: List[PoseResult] = []
        for bbox in bboxes_xyxy:
            blob, center, scale = self.preprocess(image_bgr, bbox)
            outputs = self._forward(blob)
            keypoints_batch, scores_batch = self.decode(outputs, center, scale)
            results.append(
                PoseResult(
                    keypoints_xy=keypoints_batch[0],
                    scores=scores_batch[0],
                    backend=self.name,
                    model_input_size=(self.input_width, self.input_height),
                    source_bbox_xyxy=tuple(float(v) for v in bbox),
                )
            )
        return results


class BBoxFallbackPose(PoseEstimator):
    """Deterministic pseudo-skeleton for end-to-end smoke tests only."""

    name = "bbox_fallback_not_learned"
    is_learned_model = False

    # Approximate coordinates inside an upright person box.  This verifies
    # box expansion, coordinate mapping, visualization and JSON serialization.
    _BODY_TEMPLATE = np.asarray(
        [
            (0.50, 0.08),  # nose
            (0.47, 0.07),
            (0.53, 0.07),
            (0.43, 0.09),
            (0.57, 0.09),
            (0.39, 0.23),
            (0.61, 0.23),
            (0.32, 0.39),
            (0.68, 0.39),
            (0.29, 0.55),
            (0.71, 0.55),
            (0.44, 0.52),
            (0.56, 0.52),
            (0.42, 0.71),
            (0.58, 0.71),
            (0.41, 0.91),
            (0.59, 0.91),
            (0.38, 0.98),
            (0.43, 0.98),
            (0.40, 0.94),
            (0.57, 0.98),
            (0.62, 0.98),
            (0.60, 0.94),
        ],
        dtype=np.float32,
    )

    def infer(
        self, image_bgr: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]
    ) -> List[PoseResult]:
        del image_bgr
        results: List[PoseResult] = []
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            width, height = x2 - x1, y2 - y1
            body = np.empty((23, 2), dtype=np.float32)
            body[:, 0] = x1 + self._BODY_TEMPLATE[:, 0] * width
            body[:, 1] = y1 + self._BODY_TEMPLATE[:, 1] * height
            keypoints = np.full((133, 2), -1.0, dtype=np.float32)
            scores = np.zeros((133,), dtype=np.float32)
            keypoints[:23] = body
            scores[:23] = 0.18  # intentionally below normal learned-pose confidence
            results.append(
                PoseResult(
                    keypoints_xy=keypoints,
                    scores=scores,
                    backend=self.name,
                    model_input_size=(0, 0),
                    source_bbox_xyxy=tuple(float(v) for v in bbox),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Pose interpretation
# ---------------------------------------------------------------------------


def pose_quality(
    pose: PoseResult,
    detection_score: float,
    bbox_xyxy: Sequence[float],
    keypoint_threshold: float,
) -> PoseQuality:
    """Compute a transparent quality gate from pose and detector evidence."""

    core_indices = np.asarray(
        [0, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        dtype=np.int64,
    )
    valid_indices = core_indices[core_indices < pose.scores.shape[0]]
    core_scores = pose.scores[valid_indices] if len(valid_indices) else np.zeros((0,))
    visible = core_scores >= float(keypoint_threshold)
    visible_count = int(np.count_nonzero(visible))
    total = int(len(valid_indices))
    visible_ratio = float(visible_count / total) if total else 0.0
    mean_confidence = float(np.mean(core_scores[visible])) if visible_count else 0.0
    bbox_height = float(bbox_xyxy[3] - bbox_xyxy[1])
    size_score = min(max(bbox_height / 120.0, 0.0), 1.0)

    score = (
        0.25 * min(max(float(detection_score), 0.0), 1.0)
        + 0.35 * min(max(mean_confidence, 0.0), 1.0)
        + 0.25 * visible_ratio
        + 0.15 * size_score
    )
    if pose.backend == BBoxFallbackPose.name:
        # Never let the smoke-test pseudo-pose look production-ready.
        score = min(score, 0.20)
        level = "SMOKE_TEST_ONLY"
    elif score >= 0.75 and visible_ratio >= 0.65:
        level = "HIGH"
    elif score >= 0.45 and visible_ratio >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    return PoseQuality(
        score=round(float(score), 6),
        level=level,
        visible_core_points=visible_count,
        total_core_points=total,
        visible_core_ratio=round(visible_ratio, 6),
        mean_core_confidence=round(mean_confidence, 6),
        bbox_height_px=round(bbox_height, 2),
    )


def torso_polygon(
    pose: PoseResult, threshold: float
) -> Optional[List[Tuple[float, float]]]:
    """Return shoulder/hip torso polygon when all four points are reliable."""

    indices = [5, 6, 12, 11]  # left shoulder, right shoulder, right hip, left hip
    if pose.keypoints_xy.shape[0] <= max(indices) or pose.scores.shape[0] <= max(indices):
        return None
    if any(float(pose.scores[index]) < threshold for index in indices):
        return None
    return [tuple(float(v) for v in pose.keypoints_xy[index]) for index in indices]


def extremal_legal_point_placeholder(
    pose: PoseResult, threshold: float
) -> Optional[Dict[str, Any]]:
    """Return candidate legal points, without yet assuming an attack direction.

    The perspective/offside module will later rank these points using a
    projective coordinate.  For now, JSON stores all visible legal candidates.
    """

    candidates: List[Tuple[int, float, float, float]] = []
    for index in sorted(LEGAL_BODY_INDICES):
        if index >= pose.scores.shape[0] or index >= pose.keypoints_xy.shape[0]:
            continue
        score = float(pose.scores[index])
        x, y = (float(v) for v in pose.keypoints_xy[index])
        if score >= threshold and x >= 0.0 and y >= 0.0:
            candidates.append((index, x, y, score))
    if not candidates:
        return None
    # This is not an offside decision.  It is a convenient image-space proxy
    # included for debugging until the vanishing-point module is connected.
    lowest = max(candidates, key=lambda item: item[2])
    return {
        "status": "debug_only_no_projective_ordering",
        "image_space_lowest_point": {
            "index": lowest[0],
            "name": WHOLEBODY_KEYPOINT_NAMES[lowest[0]],
            "x": round(lowest[1], 3),
            "y": round(lowest[2], 3),
            "score": round(lowest[3], 6),
        },
        "visible_legal_candidate_count": len(candidates),
    }


def keypoints_to_json(
    pose: PoseResult,
    *,
    keypoint_threshold: float,
) -> List[Dict[str, Any]]:
    keypoints: List[Dict[str, Any]] = []
    count = min(pose.keypoints_xy.shape[0], pose.scores.shape[0])
    for index in range(count):
        name = (
            WHOLEBODY_KEYPOINT_NAMES[index]
            if index < len(WHOLEBODY_KEYPOINT_NAMES)
            else f"keypoint_{index:03d}"
        )
        x, y = (float(v) for v in pose.keypoints_xy[index])
        score = float(pose.scores[index])
        visible = bool(score >= keypoint_threshold and x >= 0.0 and y >= 0.0)
        keypoints.append(
            {
                "index": index,
                "name": name,
                "x": round(x, 3),
                "y": round(y, 3),
                "score": round(score, 6),
                "visible": visible,
                "legal_for_offside_position": index in LEGAL_BODY_INDICES,
                "excluded_as_arm_or_hand": index in EXCLUDED_ARM_INDICES,
            }
        )
    return keypoints


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _line_width(image_bgr: np.ndarray) -> int:
    return max(2, int(round(min(image_bgr.shape[:2]) / 360.0)))


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    background: Tuple[int, int, int] = COLOR_TEXT_BG,
    foreground: Tuple[int, int, int] = COLOR_TEXT,
    scale: float = 0.52,
    thickness: int = 1,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(height + baseline + 3, y)
    cv2.rectangle(
        image,
        (x, y - height - baseline - 4),
        (x + width + 6, y + 2),
        background,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - baseline - 1),
        font,
        scale,
        foreground,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_detection(image: np.ndarray, detection: Detection, selected_ball: bool = False) -> None:
    width = _line_width(image)
    x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox_xyxy)
    color = {
        1: COLOR_BALL,
        2: COLOR_PLAYER,
        3: COLOR_GOALKEEPER,
    }.get(detection.label_id, COLOR_OTHER)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width, lineType=cv2.LINE_AA)

    number = detection.detection_id.rsplit("_", 1)[-1].lstrip("0") or "0"
    short_prefix = {1: "BALL*" if selected_ball else "BALL", 2: "P", 3: "GK", 4: "REF", 5: "AR", 6: "ST"}.get(
        detection.label_id, "OBJ"
    )
    caption = (
        f"{short_prefix} {detection.score:.2f}"
        if detection.label_id == BALL_CLASS_ID
        else f"{short_prefix}{number} {detection.score:.2f}"
    )
    _draw_label(
        image,
        caption,
        (x1, max(18, y1 - 3)),
        background=color,
        scale=0.45,
    )
    if detection.label_id == BALL_CLASS_ID:
        center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
        radius = max(9, width * 4)
        cv2.circle(image, center, radius, color, width, lineType=cv2.LINE_AA)
        cv2.line(
            image,
            (center[0] - radius, center[1]),
            (center[0] + radius, center[1]),
            color,
            width,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            image,
            (center[0], center[1] - radius),
            (center[0], center[1] + radius),
            color,
            width,
            lineType=cv2.LINE_AA,
        )


def draw_pose(
    image: np.ndarray,
    pose: PoseResult,
    quality: PoseQuality,
    *,
    keypoint_threshold: float,
    label_origin: Optional[Tuple[int, int]] = None,
) -> None:
    width = _line_width(image)
    keypoints = pose.keypoints_xy
    scores = pose.scores

    for start, end in BODY_SKELETON_EDGES:
        if start >= len(scores) or end >= len(scores):
            continue
        if float(scores[start]) < keypoint_threshold or float(scores[end]) < keypoint_threshold:
            continue
        p1 = tuple(int(round(v)) for v in keypoints[start])
        p2 = tuple(int(round(v)) for v in keypoints[end])
        if min(*p1, *p2) < 0:
            continue
        edge_is_excluded = start in EXCLUDED_ARM_INDICES or end in EXCLUDED_ARM_INDICES
        color = COLOR_EXCLUDED if edge_is_excluded else COLOR_LEGAL
        cv2.line(image, p1, p2, color, width, lineType=cv2.LINE_AA)

    for index in range(min(23, len(scores), len(keypoints))):
        if float(scores[index]) < keypoint_threshold:
            continue
        x, y = (int(round(v)) for v in keypoints[index])
        if x < 0 or y < 0:
            continue
        color = COLOR_LEGAL if index in LEGAL_BODY_INDICES else COLOR_EXCLUDED
        cv2.circle(image, (x, y), max(2, width + 1), color, thickness=-1, lineType=cv2.LINE_AA)

    polygon = torso_polygon(pose, keypoint_threshold)
    if polygon:
        polygon_np = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [polygon_np], isClosed=True, color=COLOR_TORSO, thickness=width)

    if label_origin is not None:
        _draw_label(
            image,
            f"Q:{quality.level} {quality.score:.2f}",
            label_origin,
            background=(225, 225, 225),
            scale=0.42,
        )


def draw_legend(image: np.ndarray, pose_backend: str) -> None:
    lines = [
        "SST + pose: yellow=legal body points, gray=arms/hands excluded",
        f"pose backend: {pose_backend}",
        "No offside line yet: camera geometry is the next module",
    ]
    y = image.shape[0] - 14 - (len(lines) - 1) * 22
    for line in lines:
        _draw_label(image, line, (12, y), scale=0.48)
        y += 22


def draw_fallback_watermark(image: np.ndarray) -> None:
    text = "SMOKE TEST - BBOX PSEUDO-POSE - NOT A LEARNED MODEL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(image.shape[:2]) / 900.0)
    thickness = max(1, _line_width(image))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, (image.shape[1] - width) // 2)
    y = max(height + 12, 34)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - height - 10),
        (x + width + 10, y + baseline + 8),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class SSTRTMWPipeline:
    """Load both models once, then process any number of frames."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        pose_estimator: PoseEstimator,
        sst_device: str = "auto",
        player_threshold: float = 0.50,
        goalkeeper_threshold: float = 0.50,
        ball_threshold: float = 0.45,
        referee_threshold: float = 0.55,
        staff_threshold: float = 0.60,
        person_nms_iou: float = 0.65,
        ball_nms_iou: float = 0.30,
        ball_top_k: int = 1,
        person_box_expansion: float = 1.00,
        keypoint_threshold: float = 0.25,
        include_other_classes: bool = True,
    ) -> None:
        self.sst_device = resolve_device(sst_device)
        start = time.perf_counter()
        self.sst_model, self.sst_metadata = build_sst_model(
            checkpoint_path, self.sst_device
        )
        self.sst_load_seconds = time.perf_counter() - start
        self.pose_estimator = pose_estimator
        self.class_thresholds = {
            1: float(ball_threshold),
            2: float(player_threshold),
            3: float(goalkeeper_threshold),
            4: float(referee_threshold),
            5: float(referee_threshold),
            6: float(staff_threshold),
        }
        self.person_nms_iou = float(person_nms_iou)
        self.ball_nms_iou = float(ball_nms_iou)
        self.ball_top_k = int(ball_top_k)
        self.person_box_expansion = float(person_box_expansion)
        self.keypoint_threshold = float(keypoint_threshold)
        self.include_other_classes = bool(include_other_classes)

    def process_image(
        self,
        input_path: str | Path,
        output_image_path: str | Path,
        output_json_path: str | Path,
        *,
        save_person_crops_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        input_path = Path(input_path).expanduser().resolve()
        output_image_path = Path(output_image_path).expanduser().resolve()
        output_json_path = Path(output_json_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV could not read image: {input_path}")
        image_height, image_width = image_bgr.shape[:2]

        with Image.open(input_path) as pil_loaded:
            image_pil = pil_loaded.convert("RGB")

        started = time.perf_counter()
        prediction = predict_frame(self.sst_model, image_pil, self.sst_device)
        detection_seconds = time.perf_counter() - started

        detections = postprocess_sst_prediction(
            prediction,
            image_width,
            image_height,
            class_thresholds=self.class_thresholds,
            person_nms_iou=self.person_nms_iou,
            ball_nms_iou=self.ball_nms_iou,
            ball_top_k=self.ball_top_k,
        )
        people = [item for item in detections if item.label_id in PERSON_CLASS_IDS]
        balls = [item for item in detections if item.label_id == BALL_CLASS_ID]
        selected_ball = balls[0] if balls else None

        expanded_boxes = [
            expand_box(
                person.bbox_xyxy,
                image_width,
                image_height,
                factor=self.person_box_expansion,
            )
            for person in people
        ]

        pose_started = time.perf_counter()
        pose_results = self.pose_estimator.infer(image_bgr, expanded_boxes)
        pose_seconds = time.perf_counter() - pose_started
        if len(pose_results) != len(people):
            raise RuntimeError(
                f"Pose backend returned {len(pose_results)} result(s) for "
                f"{len(people)} person box(es)."
            )

        crop_dir: Optional[Path] = None
        if save_person_crops_dir:
            crop_dir = Path(save_person_crops_dir).expanduser().resolve()
            crop_dir.mkdir(parents=True, exist_ok=True)

        annotated = image_bgr.copy()
        for detection in detections:
            if detection.label_id in PERSON_CLASS_IDS or detection.label_id == BALL_CLASS_ID:
                draw_detection(
                    annotated,
                    detection,
                    selected_ball=selected_ball is not None
                    and detection.detection_id == selected_ball.detection_id,
                )
            elif self.include_other_classes:
                draw_detection(annotated, detection)

        person_records: List[Dict[str, Any]] = []
        for person, expanded, pose in zip(people, expanded_boxes, pose_results):
            quality = pose_quality(
                pose,
                detection_score=person.score,
                bbox_xyxy=person.bbox_xyxy,
                keypoint_threshold=self.keypoint_threshold,
            )
            draw_pose(
                annotated,
                pose,
                quality,
                keypoint_threshold=self.keypoint_threshold,
                label_origin=None,
            )

            polygon = torso_polygon(pose, self.keypoint_threshold)
            record: Dict[str, Any] = {
                "detection_id": person.detection_id,
                "label_id": person.label_id,
                "label": person.label,
                "detector_score": round(person.score, 6),
                "bbox_xyxy": [round(value, 3) for value in person.bbox_xyxy],
                "expanded_pose_bbox_xyxy": [round(value, 3) for value in expanded],
                "pose": {
                    "backend": pose.backend,
                    "is_learned_model": bool(self.pose_estimator.is_learned_model),
                    "model_input_size_width_height": list(pose.model_input_size),
                    "keypoint_threshold": self.keypoint_threshold,
                    "quality": asdict(quality),
                    "keypoints": keypoints_to_json(
                        pose, keypoint_threshold=self.keypoint_threshold
                    ),
                    "torso_polygon_xy": (
                        [[round(x, 3), round(y, 3)] for x, y in polygon]
                        if polygon
                        else None
                    ),
                    "legal_point_debug": extremal_legal_point_placeholder(
                        pose, self.keypoint_threshold
                    ),
                    "law_modeling_note": (
                        "Elbows, wrists and hand landmarks are excluded. "
                        "Shoulder keypoints remain an approximation until "
                        "human-part segmentation is added for the armpit boundary."
                    ),
                },
            }
            person_records.append(record)

            if crop_dir is not None:
                ex1, ey1, ex2, ey2 = (int(round(value)) for value in expanded)
                crop = image_bgr[max(0, ey1) : min(image_height, ey2), max(0, ex1) : min(image_width, ex2)]
                if crop.size:
                    cv2.imwrite(str(crop_dir / f"{input_path.stem}_{person.detection_id}.jpg"), crop)

        draw_legend(annotated, self.pose_estimator.name)
        if not self.pose_estimator.is_learned_model:
            draw_fallback_watermark(annotated)
        if not cv2.imwrite(str(output_image_path), annotated):
            raise OSError(f"Failed to save output image: {output_image_path}")

        all_detection_records = [
            {
                "detection_id": item.detection_id,
                "label_id": item.label_id,
                "label": item.label,
                "score": round(item.score, 6),
                "bbox_xyxy": [round(value, 3) for value in item.bbox_xyxy],
                "selected_ball": bool(
                    selected_ball is not None
                    and item.detection_id == selected_ball.detection_id
                ),
            }
            for item in detections
        ]

        elapsed_seconds = time.perf_counter() - started
        payload: Dict[str, Any] = {
            "schema_version": "sst-rtmw-pose-v0.1",
            "input": {
                "path": str(input_path),
                "width": image_width,
                "height": image_height,
                "coordinate_system": "pixel xy, origin at top-left; boxes are xyxy",
            },
            "output": {
                "annotated_image": str(output_image_path),
                "json": str(output_json_path),
            },
            "models": {
                "sst": {
                    **self.sst_metadata,
                    "device": str(self.sst_device),
                    "load_seconds": round(self.sst_load_seconds, 4),
                },
                "pose": {
                    "backend": self.pose_estimator.name,
                    "is_learned_model": bool(self.pose_estimator.is_learned_model),
                },
            },
            "configuration": {
                "class_thresholds": {
                    CLASS_NAMES.get(key, str(key)): value
                    for key, value in self.class_thresholds.items()
                },
                "person_nms_iou": self.person_nms_iou,
                "ball_nms_iou": self.ball_nms_iou,
                "ball_top_k": self.ball_top_k,
                "person_box_expansion": self.person_box_expansion,
                "keypoint_threshold": self.keypoint_threshold,
            },
            "timing_seconds": {
                "sst_detection": round(detection_seconds, 4),
                "pose": round(pose_seconds, 4),
                "end_to_end_after_models_loaded": round(elapsed_seconds, 4),
            },
            "summary": {
                "num_detections": len(detections),
                "num_people_sent_to_pose": len(people),
                "num_ball_candidates": len(balls),
                "selected_ball_detection_id": (
                    selected_ball.detection_id if selected_ball is not None else None
                ),
                "offside_decision_available": False,
                "next_required_module": (
                    "field-line/camera calibration and projective ordering"
                ),
            },
            "detections": all_detection_records,
            "people": person_records,
            "warnings": (
                [
                    "bbox_fallback is a deterministic pseudo-skeleton used only "
                    "to smoke-test pipeline wiring; it is not a pose model."
                ]
                if not self.pose_estimator.is_learned_model
                else []
            ),
            "runtime": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
        }
        with output_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand_inputs(values: Sequence[str]) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths: List[Path] = []
    for value in values:
        expanded_value = os.path.expanduser(value)
        matches = [Path(item) for item in glob.glob(expanded_value)]
        if not matches:
            matches = [Path(expanded_value)]
        for match in matches:
            if match.is_dir():
                paths.extend(
                    child
                    for child in sorted(match.iterdir())
                    if child.is_file() and child.suffix.lower() in supported
                )
            elif match.is_file() and match.suffix.lower() in supported:
                paths.append(match)
            else:
                raise FileNotFoundError(f"Input path/glob did not resolve to an image: {value}")

    unique: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError("No supported input images were found.")
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SST object detection and top-down RTMW pose estimation."
    )
    parser.add_argument("--checkpoint", required=True, help="Trusted SST .pth checkpoint")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Image path(s), directory/directories, or shell glob(s)",
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for annotated images and JSON"
    )
    parser.add_argument(
        "--pose-backend",
        choices=("rtmw", "bbox_fallback"),
        default="rtmw",
        help="bbox_fallback is only for wiring/smoke tests",
    )
    parser.add_argument("--rtmw-model", default=None, help="Path to RTMW end2end.onnx")
    parser.add_argument("--rtmw-input-width", type=int, default=192)
    parser.add_argument("--rtmw-input-height", type=int, default=256)
    parser.add_argument("--rtmw-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--rtmw-bbox-padding",
        type=float,
        default=1.25,
        help="RTMW affine-crop padding; rtmlib-compatible default is 1.25",
    )
    parser.add_argument("--sst-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--player-threshold", type=float, default=0.50)
    parser.add_argument("--goalkeeper-threshold", type=float, default=0.50)
    parser.add_argument("--ball-threshold", type=float, default=0.45)
    parser.add_argument("--referee-threshold", type=float, default=0.55)
    parser.add_argument("--staff-threshold", type=float, default=0.60)
    parser.add_argument("--person-nms-iou", type=float, default=0.65)
    parser.add_argument("--ball-nms-iou", type=float, default=0.30)
    parser.add_argument("--ball-top-k", type=int, default=1)
    parser.add_argument(
        "--person-box-expansion",
        type=float,
        default=1.00,
        help="Extra expansion before RTMW; normally 1.0 because RTMW already pads 1.25x",
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument(
        "--save-person-crops",
        action="store_true",
        help="Save expanded person crops next to the outputs",
    )
    parser.add_argument(
        "--hide-other-classes",
        action="store_true",
        help="Do not draw referee/staff detections",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_paths = _expand_inputs(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pose_backend == "rtmw":
        if not args.rtmw_model:
            raise SystemExit(
                "--rtmw-model is required for --pose-backend rtmw. "
                "Use download_rtmw_model.py or upload/extract end2end.onnx."
            )
        pose_estimator: PoseEstimator = RTMWOpenCVDNNPose(
            args.rtmw_model,
            input_width=args.rtmw_input_width,
            input_height=args.rtmw_input_height,
            bbox_padding=args.rtmw_bbox_padding,
            device=args.rtmw_device,
        )
    else:
        pose_estimator = BBoxFallbackPose()

    print(f"Loading SST checkpoint on {args.sst_device} ...", flush=True)
    pipeline = SSTRTMWPipeline(
        checkpoint_path=args.checkpoint,
        pose_estimator=pose_estimator,
        sst_device=args.sst_device,
        player_threshold=args.player_threshold,
        goalkeeper_threshold=args.goalkeeper_threshold,
        ball_threshold=args.ball_threshold,
        referee_threshold=args.referee_threshold,
        staff_threshold=args.staff_threshold,
        person_nms_iou=args.person_nms_iou,
        ball_nms_iou=args.ball_nms_iou,
        ball_top_k=args.ball_top_k,
        person_box_expansion=args.person_box_expansion,
        keypoint_threshold=args.keypoint_threshold,
        include_other_classes=not args.hide_other_classes,
    )
    print(f"SST loaded in {pipeline.sst_load_seconds:.2f}s on {pipeline.sst_device}")
    print(f"Pose backend: {pose_estimator.name}")

    summaries: List[Dict[str, Any]] = []
    for index, input_path in enumerate(input_paths, start=1):
        output_image = output_dir / f"{input_path.stem}_sst_pose.jpg"
        output_json = output_dir / f"{input_path.stem}_sst_pose.json"
        crop_dir = output_dir / "person_crops" if args.save_person_crops else None
        print(f"[{index}/{len(input_paths)}] {input_path.name}", flush=True)
        payload = pipeline.process_image(
            input_path,
            output_image,
            output_json,
            save_person_crops_dir=crop_dir,
        )
        summary = {
            "input": str(input_path),
            "output_image": str(output_image),
            "output_json": str(output_json),
            **payload["summary"],
            **payload["timing_seconds"],
        }
        summaries.append(summary)
        print(
            "  people={num_people_sent_to_pose}, ball={num_ball_candidates}, "
            "sst={sst_detection:.2f}s, pose={pose:.2f}s, total={end_to_end_after_models_loaded:.2f}s".format(
                **summary
            )
        )

    manifest = {
        "schema_version": "sst-rtmw-pose-batch-v0.1",
        "pose_backend": pose_estimator.name,
        "is_learned_pose_model": pose_estimator.is_learned_model,
        "items": summaries,
    }
    manifest_path = output_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Batch manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
