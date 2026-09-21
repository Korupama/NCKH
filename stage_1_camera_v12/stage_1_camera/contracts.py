from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import numpy as np


class CameraStatus(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class PitchSpec:
    length_m: float = 105.0
    width_m: float = 68.0
    goal_width_m: float = 7.32
    goal_height_m: float = 2.44
    goal_area_depth_m: float = 5.5
    penalty_area_depth_m: float = 16.5
    penalty_mark_m: float = 11.0
    centre_circle_radius_m: float = 9.15
    origin: str = "center"
    x_axis: str = "goal_to_goal"
    y_axis: str = "touchline_to_touchline"
    z_axis: str = "up"
    dimension_source: str = "default_105x68"

    @property
    def x_limits(self) -> Tuple[float, float]:
        return (-self.length_m / 2.0, self.length_m / 2.0)

    @property
    def y_limits(self) -> Tuple[float, float]:
        return (-self.width_m / 2.0, self.width_m / 2.0)


@dataclass
class Distortion:
    model: str = "opencv"
    radial: List[float] = field(default_factory=list)       # k1,k2,k3,k4,k5,k6 as available
    tangential: List[float] = field(default_factory=list)   # p1,p2
    thin_prism: List[float] = field(default_factory=list)   # s1..s4

    def opencv_vector(self) -> np.ndarray:
        # OpenCV accepts [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4].
        k = list(self.radial) + [0.0] * (6 - len(self.radial))
        p = list(self.tangential) + [0.0] * (2 - len(self.tangential))
        s = list(self.thin_prism) + [0.0] * (4 - len(self.thin_prism))
        return np.asarray([k[0], k[1], p[0], p[1], k[2], k[3], k[4], k[5], *s], dtype=np.float64)


@dataclass
class CameraState:
    frame_index: int
    image_width: int
    image_height: int
    K: np.ndarray
    R_world_to_camera: np.ndarray
    camera_center_world_m: np.ndarray
    distortion: Distortion = field(default_factory=Distortion)
    pitch: PitchSpec = field(default_factory=PitchSpec)
    timestamp_sec: Optional[float] = None
    status: CameraStatus = CameraStatus.INVALID
    source: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    temporal: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    symmetry: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.2"

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.R_world_to_camera = np.asarray(self.R_world_to_camera, dtype=np.float64).reshape(3, 3)
        self.camera_center_world_m = np.asarray(self.camera_center_world_m, dtype=np.float64).reshape(3)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not np.all(np.isfinite(self.K)) or not np.all(np.isfinite(self.R_world_to_camera)):
            raise ValueError("camera matrices must be finite")

    @property
    def t_world_to_camera(self) -> np.ndarray:
        return -self.R_world_to_camera @ self.camera_center_world_m

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.column_stack([self.R_world_to_camera, self.t_world_to_camera])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "status": self.status.value,
            "image": {"width": self.image_width, "height": self.image_height, "pixel_space": "original_raw"},
            "pitch": asdict(self.pitch),
            "intrinsics": {
                "K": self.K.tolist(), "fx": float(self.K[0, 0]), "fy": float(self.K[1, 1]),
                "cx": float(self.K[0, 2]), "cy": float(self.K[1, 2]),
            },
            "distortion": asdict(self.distortion),
            "extrinsics": {
                "R_world_to_camera": self.R_world_to_camera.tolist(),
                "camera_center_world_m": self.camera_center_world_m.tolist(),
                "t_world_to_camera": self.t_world_to_camera.tolist(),
            },
            "projection": {"P": self.P.tolist()},
            "source": self.source,
            "evidence": self.evidence,
            "audit": self.audit,
            "diagnostics": self.diagnostics,
            "temporal": self.temporal,
            "uncertainty": self.uncertainty,
            "symmetry": self.symmetry,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str) -> None:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    def project_world(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        from .geometry import project_world
        return project_world(self, xyz, distort=distort)

    def world_ray(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from .geometry import pixel_to_world_ray
        return pixel_to_world_ray(self, uv)

    def intersect_pitch(self, uv: np.ndarray) -> np.ndarray:
        from .geometry import intersect_pixel_rays_with_pitch
        return intersect_pixel_rays_with_pitch(self, uv)


@dataclass
class CameraTimeline:
    states: List[CameraState]
    shot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.states.sort(key=lambda x: x.frame_index)
        if len({s.frame_index for s in self.states}) != len(self.states):
            raise ValueError("duplicate frame_index in CameraTimeline")

    def by_frame(self, frame_index: int) -> Optional[CameraState]:
        for s in self.states:
            if s.frame_index == frame_index:
                return s
        return None

    def valid_states(self) -> List[CameraState]:
        return [s for s in self.states if s.status == CameraStatus.VALID]
