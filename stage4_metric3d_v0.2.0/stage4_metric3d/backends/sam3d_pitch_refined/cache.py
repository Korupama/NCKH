from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import numpy as np

SAM3D_NATIVE_CACHE_SCHEMA = "stage4-sam3d-native-cache-1.0"

# Official SAM 3D Body MHR-70 ordering from sam_3d_body/metadata/mhr70.py.
# Hyphens are normalized to underscores for internal semantic matching.
MHR70_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_big_toe_tip", "left_small_toe_tip",
    "left_heel", "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint",
    "right_thumb_third_joint", "right_index_tip", "right_index_first_joint",
    "right_index_second_joint", "right_index_third_joint", "right_middle_tip",
    "right_middle_first_joint", "right_middle_second_joint", "right_middle_third_joint",
    "right_ring_tip", "right_ring_first_joint", "right_ring_second_joint",
    "right_ring_third_joint", "right_pinky_tip", "right_pinky_first_joint",
    "right_pinky_second_joint", "right_pinky_third_joint", "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint",
    "left_thumb_third_joint", "left_index_tip", "left_index_first_joint",
    "left_index_second_joint", "left_index_third_joint", "left_middle_tip",
    "left_middle_first_joint", "left_middle_second_joint", "left_middle_third_joint",
    "left_ring_tip", "left_ring_first_joint", "left_ring_second_joint",
    "left_ring_third_joint", "left_pinky_tip", "left_pinky_first_joint",
    "left_pinky_second_joint", "left_pinky_third_joint", "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
)
assert len(MHR70_NAMES) == 70


def normalize_mhr_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class Sam3DNativeCache:
    path: Path
    frame_indices: np.ndarray
    track_ids: tuple[str, ...]
    boxes_xyxy: np.ndarray
    skel_2d_px: np.ndarray
    skel_3d_relative_m: np.ndarray
    pred_cam_t_m: np.ndarray
    focal_length_px: np.ndarray
    valid_mask: np.ndarray
    joint_names: tuple[str, ...]
    metadata: dict

    @property
    def T(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def N(self) -> int:
        return len(self.track_ids)

    @property
    def J(self) -> int:
        return len(self.joint_names)

    @classmethod
    def load(cls, path: str | Path) -> "Sam3DNativeCache":
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SAM3D native cache not found: {p}")
        with np.load(p, allow_pickle=True) as data:
            schema = str(np.asarray(data["schema_version"]).item())
            if schema != SAM3D_NATIVE_CACHE_SCHEMA:
                raise ValueError(f"Unsupported SAM3D native cache schema: {schema}")
            frames = np.asarray(data["frame_indices"], dtype=np.int64)
            tids = tuple(str(x) for x in np.asarray(data["track_ids"]).tolist())
            boxes = np.asarray(data["boxes_xyxy"], dtype=np.float32)
            s2d = np.asarray(data["skel_2d_px"], dtype=np.float32)
            s3d = np.asarray(data["skel_3d_relative_m"], dtype=np.float32)
            cam_t = np.asarray(data["pred_cam_t_m"], dtype=np.float32)
            focal = np.asarray(data["focal_length_px"], dtype=np.float32)
            valid = np.asarray(data["valid_mask"], dtype=bool)
            names = tuple(normalize_mhr_name(x) for x in np.asarray(data["joint_names"]).tolist())
            meta_raw = str(np.asarray(data["metadata_json"]).item())
            metadata = json.loads(meta_raw) if meta_raw else {}
        obj = cls(p, frames, tids, boxes, s2d, s3d, cam_t, focal, valid, names, metadata)
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.frame_indices.ndim != 1:
            raise ValueError("frame_indices must be 1D")
        if np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if len(set(self.track_ids)) != self.N:
            raise ValueError("track_ids must be unique")
        if self.boxes_xyxy.shape != (self.T, self.N, 4):
            raise ValueError(f"boxes_xyxy shape mismatch: {self.boxes_xyxy.shape}")
        if self.skel_2d_px.shape != (self.T, self.N, self.J, 2):
            raise ValueError(f"skel_2d_px shape mismatch: {self.skel_2d_px.shape}")
        if self.skel_3d_relative_m.shape != (self.T, self.N, self.J, 3):
            raise ValueError(f"skel_3d_relative_m shape mismatch: {self.skel_3d_relative_m.shape}")
        if self.pred_cam_t_m.shape != (self.T, self.N, 3):
            raise ValueError(f"pred_cam_t_m shape mismatch: {self.pred_cam_t_m.shape}")
        if self.focal_length_px.shape != (self.T, self.N):
            raise ValueError(f"focal_length_px shape mismatch: {self.focal_length_px.shape}")
        if self.valid_mask.shape != (self.T, self.N):
            raise ValueError(f"valid_mask shape mismatch: {self.valid_mask.shape}")
        if self.J != 70:
            raise ValueError(f"Stage 4 v0.5 currently supports official SAM3D MHR70 output; got J={self.J}")
        if tuple(self.joint_names) != MHR70_NAMES:
            raise ValueError("SAM3D native joint schema does not match official MHR70 ordering")


def save_sam3d_native_cache(
    path: str | Path,
    *,
    frame_indices: Sequence[int],
    track_ids: Sequence[str],
    boxes_xyxy: np.ndarray,
    skel_2d_px: np.ndarray,
    skel_3d_relative_m: np.ndarray,
    pred_cam_t_m: np.ndarray,
    focal_length_px: np.ndarray,
    valid_mask: np.ndarray,
    joint_names: Sequence[str] = MHR70_NAMES,
    metadata: dict | None = None,
) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(normalize_mhr_name(x) for x in joint_names)
    payload = {
        "schema_version": np.array(SAM3D_NATIVE_CACHE_SCHEMA),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "track_ids": np.asarray([str(x) for x in track_ids]),
        "boxes_xyxy": np.asarray(boxes_xyxy, dtype=np.float32),
        "skel_2d_px": np.asarray(skel_2d_px, dtype=np.float32),
        "skel_3d_relative_m": np.asarray(skel_3d_relative_m, dtype=np.float32),
        "pred_cam_t_m": np.asarray(pred_cam_t_m, dtype=np.float32),
        "focal_length_px": np.asarray(focal_length_px, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "joint_names": np.asarray(names),
        "metadata_json": np.array(json.dumps(metadata or {}, ensure_ascii=True)),
    }
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(p)
    Sam3DNativeCache.load(p)
    return p
