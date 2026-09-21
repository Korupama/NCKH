from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

RAW_PIXEL_SPACE = "RAW_DISTORTED_PIXEL"
STAGE2_ROLES = ("player", "goalkeeper", "referee", "other")


@dataclass
class ReplayContext:
    schema_version: str
    video_path: str
    video_id: str
    fps: float
    frame_count: int
    image_width: int
    image_height: int
    selected_frame: int
    window_start: int
    window_end: int
    shot_start: int
    shot_end: int
    coordinate_space: str = RAW_PIXEL_SPACE
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.coordinate_space != RAW_PIXEL_SPACE:
            raise ValueError(
                f"Stage 2 requires {RAW_PIXEL_SPACE}; got {self.coordinate_space!r}. "
                "Do not mix raw/resized/undistorted coordinates silently."
            )
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not (0 <= self.shot_start <= self.window_start <= self.selected_frame <= self.window_end <= self.shot_end < self.frame_count):
            raise ValueError(
                "Expected 0 <= shot_start <= window_start <= selected_frame <= window_end "
                "<= shot_end < frame_count; got "
                f"{self.shot_start}, {self.window_start}, {self.selected_frame}, "
                f"{self.window_end}, {self.shot_end}, {self.frame_count}"
            )

    @property
    def frame_indices(self) -> List[int]:
        return list(range(self.window_start, self.window_end + 1))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayContext":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "replay_context" in data:
            data = data["replay_context"]
        known = set(cls.__dataclass_fields__.keys())
        extra = dict(data.get("extra") or {})
        for key, value in data.items():
            if key not in known:
                extra[key] = value
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("schema_version", "1.0")
        payload.setdefault("coordinate_space", RAW_PIXEL_SPACE)
        payload["extra"] = extra
        obj = cls(**payload)
        obj.validate()
        return obj


@dataclass
class TrackObservation:
    frame_index: int
    bbox_xyxy: List[float]
    detector_score: float
    frame_role: str
    role_evidence: Dict[str, float]
    physical_human_id: str
    source_detection_ids: List[str]
    pose_cache_key: Optional[str]
    pose_quality_score: Optional[float]
    pose_quality_level: Optional[str]
    association_cost: Optional[float] = None
    observation_tier: str = "primary"
    rescue_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityTrack:
    track_id: str
    role: str
    role_score: float
    role_margin: float
    role_status: str
    candidate_for_stage3: bool
    status: str
    start_frame: int
    end_frame: int
    observed_frames: int
    total_window_frames: int
    observed_ratio: float
    max_consecutive_gap: int
    identity_confidence: float
    observations: List[TrackObservation]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["observations"] = [x.to_dict() for x in self.observations]
        return d


@dataclass
class EntityTrackState:
    schema_version: str
    stage2_version: str
    replay_context: Dict[str, Any]
    status: str
    tracks: List[EntityTrack]
    selected_frame_entities: List[Dict[str, Any]]
    excluded_detections_by_frame: Dict[str, List[Dict[str, Any]]]
    auxiliary_ball_detections_by_frame: Dict[str, List[Dict[str, Any]]]
    stage3_handoff: Dict[str, Any]
    backend: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage2_version": self.stage2_version,
            "replay_context": self.replay_context,
            "status": self.status,
            "tracks": [t.to_dict() for t in self.tracks],
            "selected_frame_entities": self.selected_frame_entities,
            "excluded_detections_by_frame": self.excluded_detections_by_frame,
            "auxiliary_ball_detections_by_frame": self.auxiliary_ball_detections_by_frame,
            "stage3_handoff": self.stage3_handoff,
            "backend": self.backend,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "artifacts": self.artifacts,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
