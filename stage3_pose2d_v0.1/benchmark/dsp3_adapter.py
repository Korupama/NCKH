from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import json
import cv2
import numpy as np

from stage3_pose2d.wholebody133 import coco133_to_h36m17
from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN
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


def iter_3dsp(root: str | Path, split: str = "train") -> Iterator[DSP3Sample]:
    base = Path(root).expanduser().resolve()
    split_dir = base / split
    if not split_dir.is_dir() and base.name == split and base.is_dir():
        split_dir = base
    if not split_dir.is_dir():
        raise FileNotFoundError(f"3DSP split not found: {split_dir}")
    for shot_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
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


def run_3dsp_benchmark(
    root: str | Path,
    model_path: str | Path,
    *,
    split: str = "train",
    device: str = "cpu",
    max_samples: int | None = None,
    input_width: int = 288,
    input_height: int = 384,
) -> Dict[str, object]:
    model = RTMWOpenCVDNN(model_path, input_width=input_width, input_height=input_height, device=device)
    gt_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    samples: List[Dict[str, object]] = []
    for i, sample in enumerate(iter_3dsp(root, split)):
        if max_samples is not None and i >= int(max_samples):
            break
        image = cv2.imread(str(sample.image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        result = model.infer_one(image, [0.0, 0.0, float(w), float(h)])
        pred = coco133_to_h36m17(result.keypoints_xy)
        gt_all.append(sample.gt_h36m17)
        pred_all.append(pred)
        samples.append({
            "shot_id": sample.shot_id,
            "frame_id": sample.frame_id,
            "image": str(sample.image_path),
            "inference_diagnostics": result.inference_diagnostics,
        })
    if not gt_all:
        raise RuntimeError("No evaluable 3DSP samples found")
    summary = summarize_pdj(np.stack(pred_all), np.stack(gt_all))
    return {
        "schema_version": "stage3-3dsp-benchmark-1.0",
        "dataset": "3D Shot Posture Dataset (3DSP)",
        "split": split,
        "samples": len(gt_all),
        "model": str(Path(model_path).expanduser().resolve()),
        "model_input": [input_width, input_height],
        "metric_protocol": {
            "PDJ_threshold": 0.5,
            "normalization": "distance between GT shoulder-centre and GT hip-centre",
            "AUC_range": [0.0, 0.5],
        },
        "metrics": summary,
        "sample_manifest": samples,
    }
