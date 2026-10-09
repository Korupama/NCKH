from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import numpy as np

from .wholebody import POSE23_NAMES, WHOLEBODY_NAMES

STATE_WEIGHTS = {
    "VALID": 1.0,
    "LOW_MODEL_EVIDENCE": 0.60,
    "GEOMETRIC_OUTLIER": 0.25,
    "TEMPORAL_OUTLIER": 0.45,
    "LEFT_RIGHT_SUSPECT": 0.35,
    "TEMPORAL_IMPUTED": 0.30,
    "MISSING": 0.0,
}


@dataclass
class PoseObservation2D:
    track_id: str
    frame_index: int
    bbox_xyxy: np.ndarray
    uv23: np.ndarray
    states23: Tuple[str, ...]
    state_weights23: np.ndarray
    raw_scores23: np.ndarray
    pose_status: str
    source: Mapping[str, Any]


@dataclass
class Stage3Track:
    track_id: str
    role: Optional[str]
    identity_confidence: Optional[float]
    observations: List[PoseObservation2D]
    raw: Mapping[str, Any]

    def by_frame(self) -> Dict[int, PoseObservation2D]:
        return {o.frame_index: o for o in self.observations}


@dataclass
class Stage3State:
    path: Path
    raw: Dict[str, Any]
    replay_context: Dict[str, Any]
    selected_frame: int
    tracks: List[Stage3Track]

    def track_by_id(self) -> Dict[str, Stage3Track]:
        return {t.track_id: t for t in self.tracks}


def _parse_keypoints(records: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
    if len(records) != 133:
        raise ValueError(f"Expected 133 Stage-3 keypoints, got {len(records)}")
    uv = np.full((23, 2), np.nan, dtype=np.float64)
    states: List[str] = []
    scores = np.full(23, np.nan, dtype=np.float64)
    for i in range(23):
        rec = records[i]
        if int(rec.get("index", i)) != i:
            raise ValueError(f"WholeBody keypoint index mismatch at {i}")
        if str(rec.get("name", WHOLEBODY_NAMES[i])) != WHOLEBODY_NAMES[i]:
            raise ValueError(f"WholeBody keypoint name mismatch at {i}: {rec.get('name')}")
        x, y = rec.get("x"), rec.get("y")
        if x is not None and y is not None:
            try:
                x, y = float(x), float(y)
                if np.isfinite(x) and np.isfinite(y):
                    uv[i] = (x, y)
            except (TypeError, ValueError):
                pass
        state = str(rec.get("state", "MISSING"))
        states.append(state)
        try:
            score = rec.get("raw_model_score")
            scores[i] = float(score) if score is not None else np.nan
        except (TypeError, ValueError):
            pass
    return uv, tuple(states), scores


def load_stage3_state(path: str | Path) -> Stage3State:
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("schema_version")) != "tracked-pose-2d-state-1.0":
        raise ValueError(f"Unsupported Stage-3 schema: {data.get('schema_version')}")
    if str(data.get("coordinate_space")) != "RAW_DISTORTED_PIXEL":
        raise ValueError("Stage 4 requires Stage-3 RAW_DISTORTED_PIXEL coordinates")
    kp_schema = data.get("keypoint_schema") or {}
    if int(kp_schema.get("count", -1)) != 133:
        raise ValueError("Stage 4 requires canonical WholeBody133 Stage-3 input")
    names = kp_schema.get("names")
    if names is not None and tuple(names) != WHOLEBODY_NAMES:
        raise ValueError("Stage-3 WholeBody133 ordering does not match Stage 4")

    replay = dict(data.get("replay_context") or {})
    selected = int(replay.get("selected_frame"))
    tracks: List[Stage3Track] = []
    seen = set()
    for tr in data.get("tracks", []):
        tid = str(tr["track_id"])
        obs_out: List[PoseObservation2D] = []
        for obs in tr.get("observations", []):
            frame = int(obs["frame_index"])
            key = (tid, frame)
            if key in seen:
                raise ValueError(f"Duplicate Stage-3 observation: {tid} frame {frame}")
            seen.add(key)
            uv, states, scores = _parse_keypoints(obs.get("keypoints_133") or [])
            weights = np.asarray([STATE_WEIGHTS.get(s, 0.2) for s in states], dtype=np.float64)
            weights[~np.isfinite(uv).all(axis=1)] = 0.0
            # Preserve the raw pose for auditing, but never fit 3D or choose a
            # ground contact using a pose rejected by Stage-3 ownership QA.
            if str(obs.get("pose_status", "MISSING")) in {"REJECTED", "MISSING"}:
                weights[:] = 0.0
            bbox = np.asarray(obs.get("source_bbox_xyxy") or [np.nan] * 4, dtype=np.float64)
            obs_out.append(PoseObservation2D(
                track_id=tid,
                frame_index=frame,
                bbox_xyxy=bbox,
                uv23=uv,
                states23=states,
                state_weights23=weights,
                raw_scores23=scores,
                pose_status=str(obs.get("pose_status", "MISSING")),
                source=obs,
            ))
        tracks.append(Stage3Track(
            track_id=tid,
            role=tr.get("upstream_role"),
            identity_confidence=tr.get("upstream_identity_confidence"),
            observations=sorted(obs_out, key=lambda x: x.frame_index),
            raw=tr,
        ))
    return Stage3State(path=path, raw=data, replay_context=replay, selected_frame=selected, tracks=tracks)
