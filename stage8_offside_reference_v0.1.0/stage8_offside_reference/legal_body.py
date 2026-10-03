from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .adapters import finite_xyz, selected_stage4_observation


# Project proxy derived from the Stage-4 Pose23 contract. Arms below the shoulder
# are excluded. This is intentionally called a landmark proxy, not an exact body surface.
LEGAL_LANDMARKS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
EXCLUDED_ARM_LANDMARKS = (
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
POLICY_ID = "POSE23_LEGAL_LANDMARK_PROXY_V1"


def goalward_coordinate(x_world_m: float, s: int) -> float:
    if s not in (-1, 1):
        raise ValueError("attack direction s must be -1 or +1")
    return float(s) * float(x_world_m)


def legal_landmark_extent(
    track: Mapping[str, Any],
    frame_index: int,
    s: int,
    *,
    allow_root_fallback: bool = False,
) -> Dict[str, Any]:
    tid = str(track.get("track_id")) if track.get("track_id") is not None else None
    obs = selected_stage4_observation(track, frame_index)
    if obs is None:
        return {
            "track_id": tid,
            "status": "MISSING",
            "reason": "SELECTED_FRAME_OBSERVATION_MISSING",
            "legal_landmark_count": 0,
            "goalward_q_m": None,
            "goalward_x_m": None,
            "anchor": None,
            "selected_frame_status": track.get("selected_frame_status"),
        }

    candidates: List[Dict[str, Any]] = []
    for joint in obs.get("joints_world") or []:
        if not isinstance(joint, dict):
            continue
        name = str(joint.get("name") or joint.get("canonical_name") or "")
        if name not in LEGAL_LANDMARKS or joint.get("valid", True) is False:
            continue
        xyz = finite_xyz(joint.get("xyz_world_m"))
        if xyz is None:
            continue
        candidates.append({
            "name": name,
            "xyz_world_m": xyz,
            "x_world_m": xyz[0],
            "goalward_q_m": goalward_coordinate(xyz[0], s),
        })

    if not candidates:
        if allow_root_fallback:
            root = finite_xyz(obs.get("root_world_m") or track.get("root_world_m"))
            if root is not None:
                q_val = goalward_coordinate(root[0], s)
                return {
                    "track_id": tid,
                    "status": "DEGRADED",
                    "reason": "ROOT_WORLD_FALLBACK_ACTIVE",
                    "legal_landmark_count": 0,
                    "goalward_q_m": float(q_val),
                    "goalward_x_m": float(root[0]),
                    "anchor": {
                        "name": "root_world_fallback",
                        "xyz_world_m": [float(root[0]), float(root[1]), float(root[2])],
                    },
                    "selected_frame_status": "DEGRADED",
                    "candidate_names": ["root_world_fallback"],
                }
        return {
            "track_id": tid,
            "status": "UNUSABLE",
            "reason": "NO_FINITE_LEGAL_LANDMARK",
            "legal_landmark_count": 0,
            "goalward_q_m": None,
            "goalward_x_m": None,
            "anchor": None,
            "selected_frame_status": track.get("selected_frame_status"),
        }

    # Stable tie-break by anchor name only for reproducibility. The q value is unchanged.
    anchor = sorted(candidates, key=lambda row: (-row["goalward_q_m"], row["name"]))[0]
    upstream_status = str(track.get("selected_frame_status") or obs.get("quality") or "UNKNOWN").upper()
    if upstream_status in {"MISSING", "REJECTED", "INVALID", "FAILED"}:
        return {
            "track_id": tid,
            "status": "UNUSABLE",
            "reason": f"UPSTREAM_SELECTED_FRAME_STATUS_{upstream_status}",
            "legal_landmark_count": len(candidates),
            "goalward_q_m": None,
            "goalward_x_m": None,
            "anchor": None,
            "selected_frame_status": upstream_status,
            "candidate_names": sorted(row["name"] for row in candidates),
        }
    status = "VALID" if upstream_status == "VALID" else "DEGRADED"
    return {
        "track_id": tid,
        "status": status,
        "reason": None,
        "legal_landmark_count": len(candidates),
        "goalward_q_m": float(anchor["goalward_q_m"]),
        "goalward_x_m": float(anchor["x_world_m"]),
        "anchor": {
            "name": anchor["name"],
            "xyz_world_m": anchor["xyz_world_m"],
        },
        "selected_frame_status": upstream_status,
        "candidate_names": sorted(row["name"] for row in candidates),
    }
