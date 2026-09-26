from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import json
import hashlib
import cv2
import numpy as np

from stage3_pose2d.wholebody133 import coco133_to_h36m17
from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN
from stage3_pose2d.quality import evaluate_pose, pose_selection_score, status_rank
from stage3_pose2d.schemas import Stage3Config
from .metrics import summarize_pdj

H36M17_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck_nose", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)

_ALIAS = {
    "hip": "pelvis", "root": "pelvis", "pelvis": "pelvis",
    "rhip": "right_hip", "right_hip": "right_hip", "right hip": "right_hip",
    "rknee": "right_knee", "right_knee": "right_knee", "right knee": "right_knee",
    "rankle": "right_ankle", "right_ankle": "right_ankle", "right ankle": "right_ankle",
    "lhip": "left_hip", "left_hip": "left_hip", "left hip": "left_hip",
    "lknee": "left_knee", "left_knee": "left_knee", "left knee": "left_knee",
    "lankle": "left_ankle", "left_ankle": "left_ankle", "left ankle": "left_ankle",
    "spine": "spine", "thorax": "thorax", "chest": "thorax",
    "neck/nose": "neck_nose", "neck_nose": "neck_nose", "neck": "neck_nose", "nose": "neck_nose",
    "head": "head",
    "lshoulder": "left_shoulder", "left_shoulder": "left_shoulder", "left shoulder": "left_shoulder",
    "lelbow": "left_elbow", "left_elbow": "left_elbow", "left elbow": "left_elbow",
    "lwrist": "left_wrist", "left_wrist": "left_wrist", "left wrist": "left_wrist",
    "rshoulder": "right_shoulder", "right_shoulder": "right_shoulder", "right shoulder": "right_shoulder",
    "relbow": "right_elbow", "right_elbow": "right_elbow", "right elbow": "right_elbow",
    "rwrist": "right_wrist", "right_wrist": "right_wrist", "right wrist": "right_wrist",
}


def _norm_name(name: object) -> str:
    return str(name).strip().lower().replace("-", "_")


def parse_3dsp_keypoints_2d(data: Mapping[str, object]) -> np.ndarray:
    raw = None
    for key in ("keypont_2d", "keypoint_2d", "keypoints_2d"):
        if key in data:
            raw = data[key]
            break
    if not isinstance(raw, Mapping):
        raise ValueError("3DSP posture JSON lacks keypont_2d/keypoint_2d")
    items = list(raw.items())
    # Numeric joint ids are the authoritative H36M ordering in the public dataset.
    try:
        items.sort(key=lambda kv: int(kv[0]))
    except Exception:
        pass
    ordered = np.full((17, 2), np.nan, dtype=np.float32)
    unnamed: List[Tuple[float, float]] = []
    name_hits = 0
    for _, value in items:
        if not isinstance(value, Mapping):
            continue
        try:
            x, y = float(value["x"]), float(value["y"])
        except Exception:
            continue
        name = _ALIAS.get(_norm_name(value.get("name", "")))
        if name in H36M17_NAMES:
            ordered[H36M17_NAMES.index(name)] = (x, y)
            name_hits += 1
        unnamed.append((x, y))
    if name_hits < 10:
        if len(unnamed) != 17:
            raise ValueError(f"Expected 17 H36M joints, got {len(unnamed)}")
        ordered = np.asarray(unnamed, dtype=np.float32)
    if ordered.shape != (17, 2):
        raise ValueError(f"Bad 3DSP pose shape: {ordered.shape}")
    return ordered


@dataclass(frozen=True)
class DSP3Sample:
    shot_id: str
    frame_id: int
    image_path: Path
    posture_path: Path
    gt_h36m17: np.ndarray


def _load_shot_manifest(shot_manifest: str | Path | None) -> Optional[set[str]]:
    if shot_manifest is None:
        return None
    path = Path(shot_manifest).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        shot_ids = data
    else:
        shot_ids = data.get("shot_ids") or data.get("holdout_shots")
    if not isinstance(shot_ids, list) or not all(isinstance(x, (str, int)) for x in shot_ids):
        raise ValueError(f"Shot manifest must contain a shot_ids list: {path}")
    return {str(x) for x in shot_ids}


def iter_3dsp(
    root: str | Path,
    split: str = "train",
    *,
    shot_manifest: str | Path | None = None,
) -> Iterator[DSP3Sample]:
    base = Path(root).expanduser().resolve()
    split_dir = base / split
    if not split_dir.is_dir() and base.name == split and base.is_dir():
        split_dir = base
    if not split_dir.is_dir():
        raise FileNotFoundError(f"3DSP split not found: {split_dir}")
    allowed_shots = _load_shot_manifest(shot_manifest)
    for shot_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if allowed_shots is not None and shot_dir.name not in allowed_shots:
            continue
        img_dir, posture_dir = shot_dir / "img", shot_dir / "posture"
        if not img_dir.is_dir() or not posture_dir.is_dir():
            continue
        for posture_path in sorted(posture_dir.glob("*.json")):
            try:
                frame_id = int(posture_path.stem)
            except ValueError:
                continue
            image_path = img_dir / f"{frame_id:03d}.jpg"
            if not image_path.is_file():
                alternatives = list(img_dir.glob(f"{posture_path.stem}.*"))
                if not alternatives:
                    continue
                image_path = alternatives[0]
            data = json.loads(posture_path.read_text(encoding="utf-8"))
            yield DSP3Sample(shot_dir.name, frame_id, image_path, posture_path, parse_3dsp_keypoints_2d(data))


def inspect_3dsp(root: str | Path) -> Dict[str, object]:
    base = Path(root).expanduser().resolve()
    report: Dict[str, object] = {"root": str(base), "splits": {}}
    for split in ("train", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            continue
        shots = [p for p in split_dir.iterdir() if p.is_dir()]
        postures = sum(len(list((p / "posture").glob("*.json"))) for p in shots if (p / "posture").is_dir())
        images = sum(len(list((p / "img").glob("*"))) for p in shots if (p / "img").is_dir())
        report["splits"][split] = {"shots": len(shots), "images": images, "posture_json": postures}
    return report


def normalize_crop_scales(crop_scales: Sequence[float]) -> List[float]:
    """Validate an explicit multi-crop sweep and retain the 1.0 control."""
    values = [float(value) for value in crop_scales]
    if not values:
        raise ValueError("At least one crop scale is required")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Crop scales must be finite positive numbers")
    # Preserve user order while removing duplicates; insert the control first so
    # a quality-score tie remains anchored to the current preprocessing.
    unique = list(dict.fromkeys(values))
    return [1.0, *[value for value in unique if value != 1.0]]


def _select_crop_candidate(candidates: Sequence[Mapping[str, object]]) -> int:
    """Select by Stage-3 QA only; no ground-truth metric enters this ranking."""
    if not candidates:
        raise ValueError("No crop candidates to select")

    def rank(item: Mapping[str, object]):
        qa = item["qa"]
        scale = float(item["crop_scale"])
        return (
            status_rank(str(qa.get("pose_status", "REJECTED"))),
            pose_selection_score(qa),
            -abs(scale - 1.0),
            -scale,
        )

    return max(range(len(candidates)), key=lambda index: rank(candidates[index]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_3dsp_benchmark(
    root: str | Path,
    model_path: str | Path,
    *,
    split: str = "train",
    device: str = "cpu",
    max_samples: int | None = None,
    input_width: int = 288,
    input_height: int = 384,
    bbox_padding: float = 1.25,
    crop_scale: Optional[float] = None,
    crop_scales: Optional[Sequence[float]] = None,
    shot_manifest: str | Path | None = None,
) -> Dict[str, object]:
    if crop_scales is not None and crop_scale is not None:
        raise ValueError("Use either crop_scale or crop_scales, not both")
    scales = normalize_crop_scales(crop_scales) if crop_scales is not None else [1.0 if crop_scale is None else float(crop_scale)]
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("Crop scales must be finite positive numbers")

    model = RTMWOpenCVDNN(
        model_path,
        input_width=input_width,
        input_height=input_height,
        bbox_padding=bbox_padding,
        device=device,
    )
    gt_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    predictions_by_scale: Dict[float, List[np.ndarray]] = {scale: [] for scale in scales}
    samples: List[Dict[str, object]] = []
    qa_config = Stage3Config()
    for i, sample in enumerate(iter_3dsp(root, split, shot_manifest=shot_manifest)):
        if max_samples is not None and i >= int(max_samples):
            break
        image = cv2.imread(str(sample.image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        bbox = [0.0, 0.0, float(w), float(h)]
        candidates = []
        attempts = []
        for scale in scales:
            result = model.infer_one(image, bbox, crop_scale=scale)
            qa, _ = evaluate_pose(result.keypoints_xy, result.scores, bbox, qa_config)
            pred_candidate = coco133_to_h36m17(result.keypoints_xy)
            candidates.append({
                "crop_scale": scale,
                "qa": qa,
                "prediction": pred_candidate,
            })
            predictions_by_scale[scale].append(pred_candidate)
            attempts.append({
                "crop_scale": scale,
                "pose_status": qa.get("pose_status"),
                "selection_score": pose_selection_score(qa),
                "qa": qa,
                "inference_diagnostics": result.inference_diagnostics,
            })

        selected_index = _select_crop_candidate(candidates)
        selected = candidates[selected_index]
        pred = np.asarray(selected["prediction"], dtype=np.float32)
        gt_all.append(sample.gt_h36m17)
        pred_all.append(pred)
        samples.append({
            "shot_id": sample.shot_id,
            "frame_id": sample.frame_id,
            "image": str(sample.image_path),
            "selected_crop_scale": float(selected["crop_scale"]),
            "selection_policy": "status_rank_then_stage3_quality_score_then_nearest_to_1.0; no ground truth",
            "crop_attempts": attempts,
        })
    if not gt_all:
        raise RuntimeError("No evaluable 3DSP samples found")
    summary = summarize_pdj(np.stack(pred_all), np.stack(gt_all))
    return {
        "schema_version": "stage3-3dsp-benchmark-1.0",
        "dataset": "3D Shot Posture Dataset (3DSP)",
        "split": split,
        "shot_manifest": None if shot_manifest is None else str(Path(shot_manifest).expanduser().resolve()),
        "shot_ids": sorted({str(sample["shot_id"]) for sample in samples}),
        "samples": len(gt_all),
        "model": str(Path(model_path).expanduser().resolve()),
        "model_input": [input_width, input_height],
        "preprocessing": {
            "bbox_padding": float(bbox_padding),
            "crop_scale": scales[0] if len(scales) == 1 else None,
            "crop_scales": scales,
            "selection_policy": "Stage-3 QA only; ground truth is used only for final metrics",
            "protocol": "full-image bbox for 3DSP; crop scales are explicit ablation parameters",
        },
        "model_sha256": _sha256_file(Path(model_path).expanduser().resolve()),
        "metric_protocol": {
            "PDJ_threshold": 0.5,
            "normalization": "distance between GT shoulder-centre and GT hip-centre",
            "AUC_range": [0.0, 0.5],
        },
        "metrics": summary,
        "metrics_by_crop_scale": {
            str(scale): summarize_pdj(np.stack(predictions), np.stack(gt_all))
            for scale, predictions in predictions_by_scale.items()
        },
        "selected_crop_scale_counts": {
            str(scale): sum(float(sample["selected_crop_scale"]) == scale for sample in samples)
            for scale in scales
        },
        "sample_manifest": samples,
    }
