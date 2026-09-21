from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import json
import re
import cv2
import numpy as np


@dataclass
class CameraStateLite:
    frame_index: int
    image_width: int
    image_height: int
    K: np.ndarray
    R_world_to_camera: np.ndarray
    camera_center_world_m: np.ndarray
    distortion: np.ndarray
    status: str
    pitch: Dict[str, Any]
    schema_version: str = "1.2"
    raw: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraStateLite":
        image = data.get("image") or {}
        intr = data.get("intrinsics") or {}
        ext = data.get("extrinsics") or {}
        dist = data.get("distortion") or {}
        radial = list(dist.get("radial") or []) + [0.0] * 6
        tang = list(dist.get("tangential") or []) + [0.0] * 2
        prism = list(dist.get("thin_prism") or []) + [0.0] * 4
        opencv_dist = np.asarray([
            radial[0], radial[1], tang[0], tang[1], radial[2], radial[3], radial[4], radial[5],
            prism[0], prism[1], prism[2], prism[3],
        ], dtype=np.float64)
        return cls(
            frame_index=int(data["frame_index"]),
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            K=np.asarray(intr["K"], dtype=np.float64).reshape(3, 3),
            R_world_to_camera=np.asarray(ext["R_world_to_camera"], dtype=np.float64).reshape(3, 3),
            camera_center_world_m=np.asarray(ext["camera_center_world_m"], dtype=np.float64).reshape(3),
            distortion=opencv_dist,
            status=str(data.get("status", "INVALID")),
            pitch=dict(data.get("pitch") or {}),
            schema_version=str(data.get("schema_version", "unknown")),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraStateLite":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def t_world_to_camera(self) -> np.ndarray:
        return -self.R_world_to_camera @ self.camera_center_world_m

    def world_to_camera(self, xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=np.float64)
        one = pts.ndim == 1
        pts = pts.reshape(-1, 3)
        out = (self.R_world_to_camera @ (pts - self.camera_center_world_m).T).T
        return out[0] if one else out

    def camera_depth(self, xyz: np.ndarray) -> np.ndarray:
        cam = self.world_to_camera(xyz)
        return np.asarray(cam)[..., 2]

    def project_world(self, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
        pts = np.asarray(xyz, dtype=np.float64)
        one = pts.ndim == 1
        pts = pts.reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(self.R_world_to_camera)
        dist = self.distortion if distort else np.zeros_like(self.distortion)
        uv, _ = cv2.projectPoints(pts, rvec, self.t_world_to_camera.reshape(3, 1), self.K, dist)
        uv = uv.reshape(-1, 2)
        return uv[0] if one else uv

    def world_ray(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(uv, dtype=np.float64)
        one = pts.ndim == 1
        pts = pts.reshape(-1, 2)
        und = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.distortion, P=self.K).reshape(-1, 2)
        invK = np.linalg.inv(self.K)
        dirs_cam = (invK @ np.column_stack([und, np.ones(len(und))]).T).T
        dirs_world = (self.R_world_to_camera.T @ dirs_cam.T).T
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True).clip(min=1e-12)
        origins = np.repeat(self.camera_center_world_m[None, :], len(pts), axis=0)
        if one:
            return origins[0], dirs_world[0]
        return origins, dirs_world

    def intersect_z_plane(self, uv: np.ndarray, z_m: float = 0.0) -> np.ndarray:
        origins, dirs = self.world_ray(uv)
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        one = origins.ndim == 1
        origins = origins.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        denom = dirs[:, 2]
        lam = np.full(len(dirs), np.nan, dtype=np.float64)
        good = np.abs(denom) > 1e-10
        lam[good] = (float(z_m) - origins[good, 2]) / denom[good]
        good &= lam > 0
        out = origins + lam[:, None] * dirs
        out[~good] = np.nan
        return out[0] if one else out

    def intersect_pitch(self, uv: np.ndarray) -> np.ndarray:
        return self.intersect_z_plane(uv, 0.0)


class CameraTimelineLite:
    def __init__(self, states: Mapping[int, CameraStateLite], source_dir: str | Path):
        self.states = dict(states)
        self.source_dir = str(Path(source_dir).resolve())

    @classmethod
    def load_dir(cls, directory: str | Path) -> "CameraTimelineLite":
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Camera-state directory not found: {directory}")
        states: Dict[int, CameraStateLite] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                cam = CameraStateLite.load(path)
            except Exception:
                continue
            if cam.frame_index in states:
                raise ValueError(f"Duplicate camera frame {cam.frame_index} in {directory}")
            states[cam.frame_index] = cam
        if not states:
            raise ValueError(f"No CameraState JSON files found in {directory}")
        return cls(states, directory)

    def by_frame(self, frame_index: int) -> Optional[CameraStateLite]:
        return self.states.get(int(frame_index))

    def available_frames(self) -> Tuple[int, ...]:
        return tuple(sorted(self.states))
