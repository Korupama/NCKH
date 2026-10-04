from __future__ import annotations

"""Stage-4-owned benchmark contract for pretrained model-only experiments.

This module intentionally has no Stage-1 or Stage-3 imports. Benchmark camera
states and labels are explicit data inputs, not project-stage handoffs.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import CameraStateLite
from .wholebody import POSE23_NAMES


MODEL_ONLY_MANIFEST_SCHEMA = "stage4-model-only-benchmark-manifest-1.0"
METRIC_POSE23_GT_SCHEMA = "metric-pose23-gt-1.0"
_PLACEHOLDER_MARKERS = (
    "placeholder", "replace-me", "replace_with", "todo", "fake", "dummy",
    "record_the_exact", "record_dataset", "example.org",
)


@dataclass(frozen=True)
class ModelOnlyRecord:
    record_id: str
    sequence_id: str
    split: str
    image_path: Path
    image_width: int
    image_height: int
    bbox_xyxy: tuple[float, float, float, float]
    camera_path: Path
    camera: CameraStateLite
    gt_path: Path
    gt_xyz_world_m: np.ndarray
    gt_visible: np.ndarray
    input_hashes: dict[str, str]


@dataclass(frozen=True)
class ModelOnlyManifest:
    path: Path
    dataset: dict[str, Any]
    split: str
    coordinate_frame: dict[str, Any]
    records: tuple[ModelOnlyRecord, ...]
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_stage_dependencies(value: Any, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("_", "")
            if "stage1" in normalized or "stage3" in normalized:
                raise ValueError(f"Model-only manifest must not depend on Stage 1/3 ({path}.{key})")
            _reject_stage_dependencies(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_stage_dependencies(child, f"{path}[{index}]")


def _resolve_asset(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative or absolute path")
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(f"{label} looks like a placeholder path: {value}")
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"Empty {label}: {resolved}")
    return resolved


def _load_camera(path: Path, width: int, height: int, coordinate_frame: dict[str, Any]) -> CameraStateLite:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        camera = CameraStateLite.from_dict(raw)
    except Exception as exc:
        raise ValueError(f"Invalid benchmark camera state {path}: {exc}") from exc
    if camera.status != "VALID":
        raise ValueError(f"Benchmark camera must have status VALID: {path} ({camera.status})")
    if camera.image_width != width or camera.image_height != height:
        raise ValueError(
            f"Camera/image dimension mismatch in {path}: camera={camera.image_width}x{camera.image_height}, "
            f"image={width}x{height}"
        )
    if not np.isfinite(camera.K).all() or camera.K[0, 0] <= 0 or camera.K[1, 1] <= 0:
        raise ValueError(f"Invalid camera intrinsics in {path}")
    rotation = camera.R_world_to_camera
    if not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise ValueError(f"Camera rotation is not a finite orthonormal matrix: {path}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError(f"Camera rotation determinant must be +1: {path}")
    if not np.isfinite(camera.camera_center_world_m).all():
        raise ValueError(f"Invalid camera center in {path}")
    pitch_length = camera.pitch.get("length_m")
    pitch_width = camera.pitch.get("width_m")
    if (
        camera.pitch.get("origin") != "center"
        or not isinstance(pitch_length, (int, float))
        or not isinstance(pitch_width, (int, float))
        or not np.isfinite([pitch_length, pitch_width]).all()
        or pitch_length <= 0
        or pitch_width <= 0
    ):
        raise ValueError(f"Camera must declare pitch origin=center and positive length_m/width_m: {path}")
    declared = raw.get("coordinate_frame")
    if declared != coordinate_frame.get("name"):
        raise ValueError(
            f"Camera coordinate_frame must equal manifest coordinate_frame.name "
            f"({coordinate_frame.get('name')!r}), got {declared!r}"
        )
    return camera


def _load_gt(path: Path, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot parse metric GT JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != METRIC_POSE23_GT_SCHEMA:
        raise ValueError(f"GT must use {METRIC_POSE23_GT_SCHEMA}: {path}")
    poses = data.get("poses")
    if not isinstance(poses, list) or len(poses) != 1 or not isinstance(poses[0], dict):
        raise ValueError(f"GT must contain exactly one pose record: {path}")
    pose = poses[0]
    if str(pose.get("record_id")) != record_id:
        raise ValueError(f"GT record_id mismatch for {record_id}: {pose.get('record_id')!r}")
    if pose.get("joint_names") != list(POSE23_NAMES):
        raise ValueError(f"GT joint_names must exactly match canonical Pose23 order: {path}")
    xyz_raw = pose.get("xyz23_world_m")
    visible_raw = pose.get("visible23")
    if not isinstance(xyz_raw, list) or len(xyz_raw) != 23:
        raise ValueError(f"GT xyz23_world_m must have 23 joints: {path}")
    if not isinstance(visible_raw, list) or len(visible_raw) != 23 or any(type(v) is not bool for v in visible_raw):
        raise ValueError(f"GT visible23 must contain 23 booleans: {path}")
    try:
        xyz = np.asarray(xyz_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GT joints must contain numeric XYZ values: {path}") from exc
    visible = np.asarray(visible_raw, dtype=bool)
    if xyz.shape != (23, 3):
        raise ValueError(f"GT xyz23_world_m must have shape (23,3), got {xyz.shape}")
    if not np.isfinite(xyz[visible]).all():
        raise ValueError(f"Visible GT joints must have finite XYZ coordinates: {path}")
    xyz = xyz.copy()
    xyz[~visible] = np.nan
    return xyz, visible


def load_model_only_manifest(path: str | Path) -> ModelOnlyManifest:
    """Validate all manifest/assets before inference; paths are manifest-relative."""
    manifest_path = Path(path).expanduser().resolve()
    raw_bytes = manifest_path.read_bytes()
    try:
        data = json.loads(raw_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot parse model-only manifest: {manifest_path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != MODEL_ONLY_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported model-only manifest schema: {manifest_path}")
    _reject_stage_dependencies(data)

    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset metadata is required")
    for key in ("name", "release", "license_id"):
        value = dataset.get(key)
        if not isinstance(value, str) or not value.strip() or any(m in value.lower() for m in _PLACEHOLDER_MARKERS):
            raise ValueError(f"dataset.{key} must identify the real dataset/release/license")
    if (
        not isinstance(dataset.get("license_url"), str)
        or not dataset["license_url"].strip()
        or any(marker in dataset["license_url"].lower() for marker in _PLACEHOLDER_MARKERS)
    ):
        raise ValueError("dataset.license_url is required for provenance")

    split = data.get("split")
    if split not in {"development", "calibration", "holdout", "test"}:
        raise ValueError("split must be development, calibration, holdout, or test")
    coordinate_frame = data.get("coordinate_frame")
    if not isinstance(coordinate_frame, dict):
        raise ValueError("coordinate_frame metadata is required")
    if coordinate_frame.get("scope") != "PITCH_WORLD_METRIC" or coordinate_frame.get("units") != "m":
        raise ValueError("This model-only contract requires PITCH_WORLD_METRIC coordinates in metres")
    if not isinstance(coordinate_frame.get("name"), str) or not coordinate_frame["name"].strip():
        raise ValueError("coordinate_frame.name is required")
    axes = coordinate_frame.get("axes")
    if axes != {"x": "goal-to-goal", "y": "touchline-to-touchline", "z": "up"}:
        raise ValueError("coordinate_frame.axes must use the canonical pitch X/Y/Z definitions")
    if coordinate_frame.get("origin") != "pitch_center":
        raise ValueError("coordinate_frame.origin must be pitch_center")

    raw_records = data.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("records must be a non-empty list")
    base = manifest_path.parent
    records: list[ModelOnlyRecord] = []
    record_ids: set[str] = set()
    sequence_splits: dict[str, str] = {}
    for index, raw_record in enumerate(raw_records):
        label = f"records[{index}]"
        if not isinstance(raw_record, dict):
            raise ValueError(f"{label} must be an object")
        record_id = raw_record.get("record_id")
        sequence_id = raw_record.get("sequence_id")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in record_ids:
            raise ValueError(f"{label}.record_id must be non-empty and unique")
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            raise ValueError(f"{label}.sequence_id must be non-empty")
        if raw_record.get("split") != split:
            raise ValueError(f"{label}.split must match manifest split {split!r}")
        previous_split = sequence_splits.setdefault(sequence_id, split)
        if previous_split != split:
            raise ValueError(f"Sequence {sequence_id!r} crosses benchmark splits")
        record_ids.add(record_id)

        image_path = _resolve_asset(base, raw_record.get("image"), f"{label}.image")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image is not decodable: {image_path}")
        height, width = image.shape[:2]
        bbox_raw = raw_record.get("bbox_xyxy")
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            raise ValueError(f"{label}.bbox_xyxy must contain [x1,y1,x2,y2]")
        try:
            bbox = np.asarray(bbox_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.bbox_xyxy must be numeric") from exc
        if not np.isfinite(bbox).all():
            raise ValueError(f"{label}.bbox_xyxy must be finite")
        x1, y1, x2, y2 = bbox.tolist()
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            raise ValueError(f"{label}.bbox_xyxy is empty or outside image bounds {width}x{height}")

        camera_path = _resolve_asset(base, raw_record.get("camera"), f"{label}.camera")
        camera = _load_camera(camera_path, width, height, coordinate_frame)
        gt_path = _resolve_asset(base, raw_record.get("gt"), f"{label}.gt")
        gt_xyz, gt_visible = _load_gt(gt_path, record_id)
        input_hashes = {
            "image_sha256": sha256_file(image_path),
            "camera_sha256": sha256_file(camera_path),
            "gt_sha256": sha256_file(gt_path),
        }
        records.append(ModelOnlyRecord(
            record_id=record_id,
            sequence_id=sequence_id,
            split=split,
            image_path=image_path,
            image_width=width,
            image_height=height,
            bbox_xyxy=tuple(float(v) for v in bbox),
            camera_path=camera_path,
            camera=camera,
            gt_path=gt_path,
            gt_xyz_world_m=gt_xyz,
            gt_visible=gt_visible,
            input_hashes=input_hashes,
        ))
    return ModelOnlyManifest(
        path=manifest_path,
        dataset=dict(dataset),
        split=split,
        coordinate_frame=dict(coordinate_frame),
        records=tuple(records),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
