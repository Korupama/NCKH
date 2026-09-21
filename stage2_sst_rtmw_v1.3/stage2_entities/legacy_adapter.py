from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import json

from .consolidation import BALL_LABEL_ID, consolidate_human_detections


def _adapt_pose_record(human, legacy_person: Mapping[str, Any]) -> Dict[str, Any]:
    pose = dict(legacy_person.get("pose") or {})
    keypoints = []
    for kp in pose.get("keypoints", []):
        keypoints.append({
            "index": int(kp.get("index", len(keypoints))),
            "name": str(kp.get("name", f"kp_{len(keypoints):03d}")),
            "x": None if kp.get("x") is None else float(kp["x"]),
            "y": None if kp.get("y") is None else float(kp["y"]),
            "raw_score": float(kp.get("raw_score", kp.get("score", 0.0))),
            "visible": bool(kp.get("visible", False)),
        })
    return {
        "physical_human_id": human.physical_human_id,
        "source_detection_ids": list(human.source_detection_ids),
        "bbox_xyxy": list(human.bbox_xyxy),
        "expanded_pose_bbox_xyxy": [float(x) for x in legacy_person.get("expanded_pose_bbox_xyxy", human.bbox_xyxy)],
        "backend": str(pose.get("backend", "unknown")),
        "is_learned_model": bool(pose.get("is_learned_model", False)),
        "score_semantics": "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
        "keypoint_visibility_threshold": float(pose.get("keypoint_threshold", 1.0)),
        "quality": dict(pose.get("quality") or {}),
        "keypoints": keypoints,
        "legacy_source_detection_id": str(legacy_person.get("detection_id", "")),
    }


def adapt_legacy_sst_rtmw_json(
    path: str | Path,
    *,
    frame_index: int | None = None,
    cross_class_iou_threshold: float = 0.85,
    ambiguous_margin: float = 0.05,
) -> Dict[str, Any]:
    """Adapt a v0.4 `sst-rtmw-pose-v0.1` frame to Stage-2 perception schema.

    The legacy file already ran RTMW before cross-class consolidation. We use its
    poses only as a validation/replay cache; production Stage-2 inference performs
    consolidation before RTMW.
    """
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    temporal = payload.get("temporal_source") or {}
    if frame_index is None:
        frame_index = temporal.get("source_frame_index")
    if frame_index is None:
        raise ValueError(f"Cannot determine global frame index from {path}")

    detections = [dict(x) for x in payload.get("detections", [])]
    humans = consolidate_human_detections(
        detections,
        cross_class_iou_threshold=cross_class_iou_threshold,
        ambiguous_margin=ambiguous_margin,
    )
    legacy_people = {str(x.get("detection_id")): x for x in payload.get("people", [])}
    pose_cache: Dict[str, Dict[str, Any]] = {}
    for human in humans:
        if not human.pose_required:
            continue
        # Prefer the representative detection's pose, then any candidate member pose.
        candidates = [human.representative_detection_id] + [
            x for x in human.source_detection_ids if x != human.representative_detection_id
        ]
        person = next((legacy_people[x] for x in candidates if x in legacy_people), None)
        if person is not None:
            pose_cache[human.physical_human_id] = _adapt_pose_record(human, person)

    return {
        "frame_index": int(frame_index),
        "image_path": str(payload.get("input", {}).get("path", "")),
        "image_width": int(payload["input"]["width"]),
        "image_height": int(payload["input"]["height"]),
        "raw_detections": detections,
        "raw_low_score_detections": detections,
        "humans": [h.to_dict() for h in humans],
        "rescue_humans": [],
        "pose_cache": pose_cache,
        "ball_detections": [x for x in detections if int(x.get("label_id", -1)) == BALL_LABEL_ID],
        "timing_seconds": dict(payload.get("timing_seconds") or {}),
        "provenance": {
            "backend": "legacy-v04-sst-rtmw-adapter",
            "legacy_json": str(path),
            "cross_class_consolidation_before_pose": False,
            "warning": "Legacy RTMW was run before consolidation; use only for migration/tests, not timing claims.",
            "coordinate_space": "RAW_DISTORTED_PIXEL",
        },
    }


def build_manifest_from_legacy_jsons(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    cross_class_iou_threshold: float = 0.85,
    ambiguous_margin: float = 0.05,
) -> Dict[str, Any]:
    output = Path(output_dir)
    frame_dir = output / "frame_perception"
    frame_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in paths:
        adapted = adapt_legacy_sst_rtmw_json(
            source,
            cross_class_iou_threshold=cross_class_iou_threshold,
            ambiguous_margin=ambiguous_margin,
        )
        fi = int(adapted["frame_index"])
        target = frame_dir / f"frame_{fi:09d}_perception.json"
        target.write_text(json.dumps(adapted, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append({"frame_index": fi, "perception_json": str(target.resolve())})
    records.sort(key=lambda x: x["frame_index"])
    manifest = {
        "schema_version": "stage2-sst-rtmw-perception-manifest-1.0",
        "source": "legacy-v04-adapter",
        "stage2_version": "stage2-sst-rtmw-1.3.0-legacy-adapter",
        "frames": records,
        "configuration": {
            "cross_class_iou_threshold": cross_class_iou_threshold,
            "role_ambiguous_margin": ambiguous_margin,
        },
    }
    target = output / "perception_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
