"""Read-only QA for the selected-frame Stage 3 -> Stage 4 bridge."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .stage3_adapter import Stage3State
from .wholebody import LEGAL_GEOMETRY_CANDIDATE_23, POSE23_NAMES


CORE_METRIC_ANCHOR_NAMES: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_CORE_METRIC_ANCHOR_INDICES = tuple(POSE23_NAMES.index(name) for name in CORE_METRIC_ANCHOR_NAMES)


def _valid_joint_mask(uv23: np.ndarray, states23: Sequence[str]) -> np.ndarray:
    """Return raw Stage-3 joint availability without any temporal filling."""

    return np.isfinite(np.asarray(uv23, dtype=np.float64)).all(axis=1) & np.asarray(
        [state != "MISSING" for state in states23], dtype=bool
    )


def _finite_bbox(bbox_xyxy: np.ndarray) -> np.ndarray | None:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return bbox


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def audit_stage3_bridge(
    state: Stage3State,
    *,
    edge_margin_px: float = 16.0,
    crowded_iou_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Audit selected-frame evidence without changing Stage 3 or Stage 4 status.

    Thresholds identify candidates for review.  They intentionally do not
    reject a track or alter optimizer inputs.
    """

    image_width = int(state.replay_context.get("image_width", -1))
    image_height = int(state.replay_context.get("image_height", -1))
    selected = []
    for track in state.tracks:
        observation = track.by_frame().get(state.selected_frame)
        if observation is not None:
            selected.append((track, observation))

    crowded_neighbors: Dict[str, List[Dict[str, Any]]] = {track.track_id: [] for track, _ in selected}
    crowded_pairs: List[Dict[str, Any]] = []
    for index, (left_track, left_obs) in enumerate(selected):
        left_bbox = _finite_bbox(left_obs.bbox_xyxy)
        if left_bbox is None:
            continue
        for right_track, right_obs in selected[index + 1 :]:
            right_bbox = _finite_bbox(right_obs.bbox_xyxy)
            if right_bbox is None:
                continue
            iou = _bbox_iou(left_bbox, right_bbox)
            if iou < crowded_iou_threshold:
                continue
            pair = {"track_a": left_track.track_id, "track_b": right_track.track_id, "bbox_iou": iou}
            crowded_pairs.append(pair)
            crowded_neighbors[left_track.track_id].append({"track_id": right_track.track_id, "bbox_iou": iou})
            crowded_neighbors[right_track.track_id].append({"track_id": left_track.track_id, "bbox_iou": iou})

    tracks: List[Dict[str, Any]] = []
    for track, observation in selected:
        raw_valid = _valid_joint_mask(observation.uv23, observation.states23)
        core_valid = raw_valid[list(_CORE_METRIC_ANCHOR_INDICES)]
        missing_core = [name for name, is_valid in zip(CORE_METRIC_ANCHOR_NAMES, core_valid) if not is_valid]
        bbox = _finite_bbox(observation.bbox_xyxy)
        edge_risk = bool(
            bbox is not None
            and (
                bbox[0] <= edge_margin_px
                or bbox[1] <= edge_margin_px
                or bbox[2] >= image_width - edge_margin_px
                or bbox[3] >= image_height - edge_margin_px
            )
        )
        raw_pose_missing = bool(observation.pose_status == "MISSING" or not np.any(raw_valid))
        crowded = bool(crowded_neighbors[track.track_id])
        flags: List[str] = []
        if raw_pose_missing:
            flags.append("RAW_POSE_MISSING_AT_T0")
        if missing_core:
            flags.append("CORE_METRIC_ANCHOR_INCOMPLETE")
        if edge_risk:
            flags.append("FRAME_EDGE_RISK")
        if crowded:
            flags.append("CROWDED_OR_OCCLUDED_RISK")
        tracks.append(
            {
                "track_id": track.track_id,
                "stage3_pose_status": observation.pose_status,
                "source_bbox_xyxy": None if bbox is None else [float(value) for value in bbox],
                "raw_valid_joint_count_23": int(np.count_nonzero(raw_valid)),
                "raw_valid_legal_joint_count": int(
                    np.count_nonzero(raw_valid & np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool))
                ),
                "raw_valid_core_anchor_count": int(np.count_nonzero(core_valid)),
                "missing_core_anchor_names": missing_core,
                "raw_pose_missing": raw_pose_missing,
                "edge_risk": edge_risk,
                "crowded_risk": crowded,
                "crowded_neighbors": crowded_neighbors[track.track_id],
                "risk_flags": flags,
            }
        )

    def flagged(flag: str) -> List[str]:
        return [track["track_id"] for track in tracks if flag in track["risk_flags"]]

    return {
        "schema_version": "stage3-bridge-qa-1.0",
        "selected_frame": state.selected_frame,
        "coordinate_space": state.raw.get("coordinate_space"),
        "image_size": [image_width, image_height],
        "thresholds": {
            "edge_margin_px": float(edge_margin_px),
            "crowded_iou_threshold": float(crowded_iou_threshold),
            "core_metric_anchor_names": list(CORE_METRIC_ANCHOR_NAMES),
        },
        "summary": {
            "tracks_total": len(state.tracks),
            "selected_frame_observations": len(tracks),
            "raw_pose_missing_tracks": flagged("RAW_POSE_MISSING_AT_T0"),
            "edge_risk_tracks": flagged("FRAME_EDGE_RISK"),
            "crowded_risk_tracks": flagged("CROWDED_OR_OCCLUDED_RISK"),
            "incomplete_core_anchor_tracks": flagged("CORE_METRIC_ANCHOR_INCOMPLETE"),
            "crowded_pairs": len(crowded_pairs),
        },
        "crowded_pairs": crowded_pairs,
        "tracks": tracks,
        "policy": "read_only_audit; flags do not alter Stage 3 pixels, Stage 4 optimizer inputs, or acceptance status",
    }
