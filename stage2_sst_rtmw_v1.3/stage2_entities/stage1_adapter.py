from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import csv
import json
import cv2

from .contracts import ReplayContext, RAW_PIXEL_SPACE


class Stage1CompatibilityError(RuntimeError):
    pass


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise Stage1CompatibilityError(f"Required Stage-1 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Stage1CompatibilityError(f"Cannot open Stage-1 video: {video_path}")
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if min(meta["frame_count"], meta["width"], meta["height"]) <= 0 or meta["fps"] <= 0:
        raise Stage1CompatibilityError(f"Invalid video metadata: {meta}")
    return meta


def _derive_shot_bounds(path: Path, shot_id: int, frame_count: int) -> Tuple[int, int, str]:
    if not path.is_file():
        return -1, -1, "unavailable"
    frames = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                left, right = int(row["left_frame"]), int(row["right_frame"])
                before, after = int(row["shot_before"]), int(row["shot_after"])
            except Exception:
                continue
            if before == shot_id:
                frames.add(left)
            if after == shot_id:
                frames.add(right)
    if not frames:
        return -1, -1, "unavailable"
    return max(0, min(frames)), min(frame_count - 1, max(frames)), "stage1_batch_shot_transitions"


def replay_context_from_stage1_workspace(
    stage1_root: str | Path,
    *,
    video_override: Optional[str | Path] = None,
) -> ReplayContext:
    """Adapt the actual frozen Stage-1 v12 workspace to the Stage-2 ReplayContext."""
    root = Path(stage1_root).expanduser().resolve()
    if not (root / "stage_1_camera").is_dir():
        raise Stage1CompatibilityError(f"Not a Stage-1 v12 workspace: {root}")

    final = root / "outputs" / "stage1_final"
    temporal = root / "outputs" / "temporal_v13" / "batch_video"
    freeze_path = final / "CAMERA_CONTRACT_FREEZE_v12.json"
    manifest_path = final / "camera_timeline_target_window_manifest.json"
    ptz_path = temporal / "shot_ptz" / "batch_ptz_summary.json"
    transitions_path = temporal / "batch_shot_transitions.csv"

    freeze = _load_json(freeze_path)
    manifest = _load_json(manifest_path)
    ptz = _load_json(ptz_path)

    if not bool(freeze.get("stage1_ready_for_stage2")):
        raise Stage1CompatibilityError("Stage 1 does not declare stage1_ready_for_stage2=true")
    if str(freeze.get("package_version")) != "0.12.0":
        raise Stage1CompatibilityError(f"Expected Stage-1 package 0.12.0, got {freeze.get('package_version')}")
    if str(freeze.get("camera_state_schema")) != "1.2":
        raise Stage1CompatibilityError(f"Expected CameraState schema 1.2, got {freeze.get('camera_state_schema')}")

    frames = manifest.get("frames") or []
    indices = [int(x["frame_index"]) for x in frames]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise Stage1CompatibilityError("Stage-1 target-window manifest must be non-empty and contiguous")
    selected = int(manifest["target_frame_index"])
    if selected not in indices:
        raise Stage1CompatibilityError("Stage-1 selected frame is outside target window")
    if int(freeze["target_frame_index"]) != selected or int(ptz["target_frame_index"]) != selected:
        raise Stage1CompatibilityError("Stage-1 target frame disagrees across frozen artifacts")

    if video_override is not None:
        video = Path(video_override).expanduser().resolve()
    else:
        video = root / "download.mp4"
        if not video.is_file():
            raw = ptz.get("video")
            if raw and Path(raw).is_file():
                video = Path(raw).resolve()
    if not video.is_file():
        raise Stage1CompatibilityError("Original replay video not found; pass video_override explicitly")

    actual = _video_metadata(video)
    reported = ptz.get("video_metadata") or {}
    for k in ("frame_count", "width", "height"):
        if k in reported and int(reported[k]) != int(actual[k]):
            raise Stage1CompatibilityError(f"Stage-1 video metadata mismatch for {k}")
    if "fps" in reported and abs(float(reported["fps"]) - actual["fps"]) > 1e-3:
        raise Stage1CompatibilityError("Stage-1 FPS disagrees with original video")

    t0_path = final / f"camera_state_t0_{selected:08d}_with_uncertainty.json"
    t0 = _load_json(t0_path)
    image = t0.get("image") or {}
    if image.get("pixel_space") != "original_raw":
        raise Stage1CompatibilityError(
            f"Expected Stage-1 original_raw pixel space, got {image.get('pixel_space')!r}"
        )
    if int(image.get("width", -1)) != actual["width"] or int(image.get("height", -1)) != actual["height"]:
        raise Stage1CompatibilityError("Stage-1 CameraState dimensions disagree with video")

    shot_id = int(manifest.get("shot_id", 0))
    shot_start, shot_end, shot_source = _derive_shot_bounds(transitions_path, shot_id, actual["frame_count"])
    if shot_start < 0:
        shot_start, shot_end = indices[0], indices[-1]
        shot_source = "analysis_window_conservative_fallback"
    if not (shot_start <= indices[0] <= selected <= indices[-1] <= shot_end):
        raise Stage1CompatibilityError("Derived shot bounds do not contain Stage-1 analysis window")

    ctx = ReplayContext(
        schema_version="1.0-stage1-v12-adapter",
        video_path=str(video.resolve()),
        video_id=video.stem,
        fps=actual["fps"],
        frame_count=actual["frame_count"],
        image_width=actual["width"],
        image_height=actual["height"],
        selected_frame=selected,
        window_start=indices[0],
        window_end=indices[-1],
        shot_start=shot_start,
        shot_end=shot_end,
        coordinate_space=RAW_PIXEL_SPACE,
        extra={
            "source": "stage1_camera_v12_adapter",
            "stage1_root": str(root),
            "stage1_package_version": freeze.get("package_version"),
            "stage1_camera_state_schema": freeze.get("camera_state_schema"),
            "stage1_contract_freeze": str(freeze_path),
            "stage1_timeline_manifest": str(manifest_path),
            "stage1_ptz_summary": str(ptz_path),
            "stage1_optimized_camera_states_dir": str(temporal / "shot_ptz" / "optimized_camera_states"),
            "stage1_shot_id": shot_id,
            "shot_bounds_source": shot_source,
            "source_pixel_space": "original_raw",
            "pixel_space_mapping": "original_raw -> RAW_DISTORTED_PIXEL (semantic alias; no transform)",
        },
    )
    ctx.validate()
    return ctx
