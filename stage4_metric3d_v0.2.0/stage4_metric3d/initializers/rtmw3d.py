from __future__ import annotations

"""Official MMPose RTMW3D worker helpers.

This module intentionally lives behind optional imports. The Stage-4 core only
consumes the generated JSONL cache and therefore does not require OpenMMLab.

Important contract:
- Stage-3 validated 2D keypoints remain the 2D source of truth.
- RTMW3D x/y are retained only as raw provenance.
- RTMW3D z is treated as a *relative depth prior*, never global world XYZ.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import json
import os
import sys
import numpy as np

from ..stage3_adapter import Stage3State, load_stage3_state
from .cache import write_jsonl


@dataclass(frozen=True)
class RTMW3DRunConfig:
    model_config: str
    checkpoint: str
    device: str = "cpu"
    window_radius_frames: int = 13
    selected_only: bool = False


def _load_mmpose(mmpose_root: str | Path | None = None):
    """Load MMPose and the RTMPose3D project module.

    RTMW3D in MMPose v1.3.2 is implemented as an independent project. Its
    config contains ``custom_imports = ['rtmpose3d']``; therefore installing
    the base ``mmpose`` package alone is not sufficient unless
    ``projects/rtmpose3d`` is also on ``PYTHONPATH``.

    ``mmpose_root`` (or the MMPOSE_ROOT environment variable) lets the worker
    add both the repository root and ``projects/rtmpose3d`` to ``sys.path``
    explicitly, which is much less error-prone on Windows.
    """
    root = mmpose_root or os.environ.get("MMPOSE_ROOT")
    if root:
        root_path = Path(root).expanduser().resolve()
        project_path = root_path / "projects" / "rtmpose3d"
        if not root_path.exists():
            raise RuntimeError(f"MMPose root does not exist: {root_path}")
        if not project_path.exists():
            raise RuntimeError(
                f"RTMPose3D project directory not found: {project_path}. "
                "Use the official MMPose v1.3.2 source tree."
            )
        for entry in (str(root_path), str(project_path)):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    try:
        from mmpose.apis import inference_topdown, init_model  # type: ignore
        from mmpose.utils import register_all_modules  # type: ignore
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        raise RuntimeError(
            "RTMW3D worker requires a dedicated OpenMMLab environment. "
            "Install torch, mmengine, mmcv and MMPose v1.3.2, then run this "
            "worker from that environment. See setup_rtmw3d_windows.ps1."
        ) from exc

    try:
        __import__("rtmpose3d")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "MMPose is installed, but the RTMPose3D project module is not "
            "importable. RTMW3D v1.3.2 needs the official "
            "mmpose/projects/rtmpose3d directory on PYTHONPATH. Pass "
            "--mmpose-root D:\\path\\to\\mmpose or set MMPOSE_ROOT."
        ) from exc

    register_all_modules()
    return init_model, inference_topdown


def _open_video(video_path: str | Path):
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required to read replay frames") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def _read_frame(cap, frame_index: int) -> np.ndarray:
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def _wanted_observations(state: Stage3State, radius: int, selected_only: bool):
    lo = state.selected_frame if selected_only else state.selected_frame - int(radius)
    hi = state.selected_frame if selected_only else state.selected_frame + int(radius)
    by_frame: Dict[int, List[Tuple[str, object]]] = {}
    for track in state.tracks:
        for obs in track.observations:
            if lo <= obs.frame_index <= hi:
                by_frame.setdefault(obs.frame_index, []).append((track.track_id, obs))
    return by_frame


def _extract_prediction(sample) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw model keypoints (133,3) and scores (133,).

    MMPose PoseDataSample shapes vary slightly across versions; this adapter is
    intentionally defensive. No coordinate-axis swap is applied here because
    the Stage-4 optimizer only consumes the raw third channel as signed relative
    depth evidence. The model's x/y are not used for metric reconstruction.
    """
    inst = getattr(sample, "pred_instances", None)
    if inst is None:
        raise ValueError("MMPose result has no pred_instances")
    kpts = np.asarray(getattr(inst, "keypoints"), dtype=np.float64)
    if kpts.ndim == 3 and kpts.shape[0] == 1:
        kpts = kpts[0]
    if kpts.shape != (133, 3):
        raise ValueError(f"Expected RTMW3D keypoints (133,3), got {kpts.shape}")

    scores_obj = getattr(inst, "keypoint_scores", None)
    if scores_obj is None:
        scores = np.ones(133, dtype=np.float64)
    else:
        scores = np.asarray(scores_obj, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        scores = scores.reshape(-1)
        if scores.shape[0] != 133:
            raise ValueError(f"Expected 133 RTMW3D keypoint scores, got {scores.shape}")
    return kpts, scores


def generate_rtmw3d_cache(
    *,
    stage3_state: str | Path,
    video_path: str | Path,
    output_jsonl: str | Path,
    model_config: str | Path,
    checkpoint: str | Path,
    device: str = "cpu",
    window_radius_frames: int = 13,
    selected_only: bool = False,
    mmpose_root: str | Path | None = None,
) -> Path:
    """Run official RTMW3D top-down inference and cache raw evidence.

    Bounding boxes come from Stage 3's inherited Stage-2 track observations.
    The output is intentionally model-space evidence; it is not declared to be
    pitch-world metric coordinates.
    """
    init_model, inference_topdown = _load_mmpose(mmpose_root)
    state = load_stage3_state(stage3_state)
    model_config = str(Path(model_config).expanduser().resolve())
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    model = init_model(model_config, checkpoint, device=device)

    by_frame = _wanted_observations(state, window_radius_frames, selected_only)
    cap = _open_video(video_path)
    records: List[Mapping[str, object]] = []
    try:
        for frame_index in sorted(by_frame):
            frame = _read_frame(cap, frame_index)
            pairs = by_frame[frame_index]
            bboxes = np.stack([obs.bbox_xyxy for _, obs in pairs], axis=0).astype(np.float32)
            results = inference_topdown(model, frame, bboxes=bboxes, bbox_format="xyxy")
            if len(results) != len(pairs):
                raise RuntimeError(
                    f"RTMW3D result count mismatch at frame {frame_index}: "
                    f"{len(results)} vs {len(pairs)}"
                )
            for (track_id, obs), sample in zip(pairs, results):
                kpts, scores = _extract_prediction(sample)
                root_z = float(np.nanmean(kpts[[11, 12], 2]))
                rel_z = kpts[:, 2] - root_z
                records.append({
                    "record_type": "observation",
                    "track_id": track_id,
                    "frame_index": int(frame_index),
                    "backend": "RTMW3D-L-384x288_MMPose",
                    "raw_keypoints_133": kpts.tolist(),
                    "keypoint_scores_133": scores.tolist(),
                    "relative_depth_133": rel_z.tolist(),
                    "root_definition": "mean(left_hip,right_hip)",
                    "source_bbox_xyxy": np.asarray(obs.bbox_xyxy, dtype=float).tolist(),
                    "global_position_known": False,
                    "xy_used_by_stage4": False,
                    "depth_semantics": "raw_model_third_channel_centered_on_hip_root_relative_prior_only",
                })
    finally:
        cap.release()

    manifest = {
        "schema_version": "stage4-relative3d-initializer-cache-1.0",
        "backend": "RTMW3D-L-384x288_MMPose",
        "stage3_state": str(Path(stage3_state).expanduser().resolve()),
        "video": str(Path(video_path).expanduser().resolve()),
        "model_config": model_config,
        "checkpoint": checkpoint,
        "device": device,
        "selected_frame": state.selected_frame,
        "window_radius_frames": int(window_radius_frames),
        "selected_only": bool(selected_only),
        "mmpose_root": str(Path(mmpose_root).expanduser().resolve()) if mmpose_root else os.environ.get("MMPOSE_ROOT"),
        "records": len(records),
        "global_metric_position_trusted": False,
        "stage3_2d_remains_source_of_truth": True,
    }
    return write_jsonl(output_jsonl, manifest, records)
