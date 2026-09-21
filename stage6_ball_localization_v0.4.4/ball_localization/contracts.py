from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

@dataclass
class BallCandidate2D:
    frame_index: int
    candidate_id: str
    bbox_xyxy: List[float]
    center_uv: List[float]
    detector_score: float
    source: str
    diameter_px: float
    pitch_prior: float = 1.0
    ranking_score: float = 0.0
    coordinate_space: str = "RAW_DISTORTED_PIXEL"
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if self.ranking_score <= 0:
            self.ranking_score = float(self.detector_score) * float(self.pitch_prior)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class BallFrameState:
    frame_index: int
    timestamp_sec: float
    candidate: Optional[BallCandidate2D]
    observation_status: str
    camera_status: str
    localization_status: str
    ground_contact_xyz_world_m: Optional[List[float]]
    ground_center_xyz_world_m: Optional[List[float]]
    size_prior_xyz_world_m: Optional[List[float]]
    selected_center_xyz_world_m: Optional[List[float]]
    selected_method: Optional[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        d=asdict(self)
        if self.candidate is not None: d["candidate"]=self.candidate.to_dict()
        return d

@dataclass
class BallTrajectoryState:
    schema_version: str
    stage6_version: str
    replay_context: Dict[str, Any]
    detector: Dict[str, Any]
    tracker: Dict[str, Any]
    ball_radius_m: float
    localization_mode: str
    status: str
    candidates_by_frame: Dict[str, List[Dict[str, Any]]]
    frames: List[BallFrameState]
    selected_frame_ball: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version":self.schema_version,"stage6_version":self.stage6_version,"replay_context":self.replay_context,"detector":self.detector,"tracker":self.tracker,"ball_radius_m":self.ball_radius_m,"localization_mode":self.localization_mode,"status":self.status,"candidates_by_frame":self.candidates_by_frame,"frames":[x.to_dict() for x in self.frames],"selected_frame_ball":self.selected_frame_ball,"diagnostics":self.diagnostics,"artifacts":self.artifacts}
    def save_json(self,path: str|Path)->Path:
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(self.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8"); return p
