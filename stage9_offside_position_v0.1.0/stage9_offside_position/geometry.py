from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .adapters import finite_number, finite_xyz, selected_stage4_observation
from .constants import LEGAL_LANDMARK_SET


def goalward_q(x_world_m: float, s: int) -> float:
    if s not in (-1, 1):
        raise ValueError("attack direction s must be -1 or +1")
    return float(s) * float(x_world_m)


def _joint_name(joint: Mapping[str, Any]) -> Optional[str]:
    return joint.get("name") or joint.get("canonical_name")


def legal_extent_from_stage4_track(track: Mapping[str, Any], frame_index: int, s: int) -> Dict[str, Any]:
    obs = selected_stage4_observation(track, frame_index)
    if obs is None:
        return {
            "status": "MISSING",
            "source": None,
            "goalward_q_m": None,
            "goalward_x_m": None,
            "anchor": None,
            "legal_landmark_count": 0,
        }
    joints = obs.get("joints_world") or obs.get("joints") or obs.get("metric_pose23") or []
    candidates: List[Tuple[float, str, List[float]]] = []
    for joint in joints:
        if not isinstance(joint, Mapping):
            continue
        name = _joint_name(joint)
        if name not in LEGAL_LANDMARK_SET:
            continue
        if joint.get("valid") is False:
            continue
        xyz = finite_xyz(joint.get("xyz_world_m"))
        if xyz is None:
            continue
        candidates.append((goalward_q(xyz[0], s), str(name), xyz))
    if candidates:
        q, name, xyz = max(candidates, key=lambda row: row[0])
        return {
            "status": "VALID",
            "source": "STAGE4_LEGAL_LANDMARK",
            "goalward_q_m": float(q),
            "goalward_x_m": float(xyz[0]),
            "anchor": {"name": name, "xyz_world_m": xyz},
            "legal_landmark_count": len(candidates),
        }

    root = finite_xyz(obs.get("root_world_m"))
    if root is None:
        root = finite_xyz(track.get("root_world_m"))
    if root is not None:
        return {
            "status": "FALLBACK",
            "source": "STAGE4_ROOT_WORLD_FALLBACK",
            "goalward_q_m": goalward_q(root[0], s),
            "goalward_x_m": float(root[0]),
            "anchor": {"name": "root_world_m", "xyz_world_m": root},
            "legal_landmark_count": 0,
        }
    return {
        "status": "MISSING",
        "source": None,
        "goalward_q_m": None,
        "goalward_x_m": None,
        "anchor": None,
        "legal_landmark_count": 0,
    }


def stage4_world_points(track: Mapping[str, Any], frame_index: int) -> List[Dict[str, Any]]:
    obs = selected_stage4_observation(track, frame_index)
    if obs is None:
        return []
    joints = obs.get("joints_world") or obs.get("joints") or obs.get("metric_pose23") or []
    out: List[Dict[str, Any]] = []
    for joint in joints:
        if not isinstance(joint, Mapping):
            continue
        xyz = finite_xyz(joint.get("xyz_world_m"))
        if xyz is None:
            continue
        out.append({
            "name": _joint_name(joint) or "joint",
            "xyz_world_m": xyz,
            "valid": joint.get("valid", True) is not False,
        })
    return out


def reference_from_stage8_best_effort(stage8: Mapping[str, Any], s: Optional[int]) -> Dict[str, Any]:
    reference = stage8.get("reference") if isinstance(stage8.get("reference"), Mapping) else {}
    q = reference.get("goalward_q_m")
    x = reference.get("X_world_m")
    if finite_number(q):
        qf = float(q)
        xf = float(x) if finite_number(x) else (float(s) * qf if s in (-1, 1) else None)
        return {"status": "DIRECT", "goalward_q_m": qf, "X_world_m": xf, "source": reference.get("source") or "STAGE8_REFERENCE"}

    candidates: List[Tuple[float, str]] = []
    second = stage8.get("second_last_opponent") if isinstance(stage8.get("second_last_opponent"), Mapping) else {}
    ball = stage8.get("ball") if isinstance(stage8.get("ball"), Mapping) else {}
    if finite_number(second.get("goalward_q_m")):
        candidates.append((float(second["goalward_q_m"]), "SECOND_LAST_OPPONENT"))
    if finite_number(ball.get("goalward_q_m")):
        candidates.append((float(ball["goalward_q_m"]), "BALL"))
    if candidates:
        qf, source = max(candidates, key=lambda r: r[0])
        xf = float(s) * qf if s in (-1, 1) else None
        return {"status": "RECONSTRUCTED", "goalward_q_m": qf, "X_world_m": xf, "source": source}

    ranking = [r for r in (stage8.get("opponent_ranking") or []) if isinstance(r, Mapping) and finite_number(r.get("goalward_q_m"))]
    if len(ranking) >= 2:
        ranked = sorted(ranking, key=lambda r: float(r["goalward_q_m"]), reverse=True)
        qf = float(ranked[1]["goalward_q_m"])
        xf = float(s) * qf if s in (-1, 1) else None
        return {"status": "RANKING_ONLY_FALLBACK", "goalward_q_m": qf, "X_world_m": xf, "source": "SECOND_LAST_OPPONENT_FALLBACK"}
    return {"status": "MISSING", "goalward_q_m": None, "X_world_m": None, "source": None}
