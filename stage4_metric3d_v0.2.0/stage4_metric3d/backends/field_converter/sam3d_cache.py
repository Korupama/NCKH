from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import numpy as np

from .contracts import FC_EXPECTED_JOINTS, SAM3D_CACHE_SCHEMA


@dataclass(frozen=True)
class Sam3DCache:
    path: Path
    frame_indices: np.ndarray
    track_ids: tuple[str, ...]
    boxes_xyxy: np.ndarray
    skel_2d_px: np.ndarray
    skel_3d_relative_m: np.ndarray
    valid_mask: np.ndarray
    joint_names: tuple[str, ...]
    semantic_mapping_validated: bool
    metadata: dict

    @property
    def T(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def N(self) -> int:
        return len(self.track_ids)

    @classmethod
    def load(cls, path: str | Path) -> "Sam3DCache":
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SAM3D cache not found: {p}")
        with np.load(p, allow_pickle=True) as data:
            schema = str(np.asarray(data["schema_version"]).item())
            if schema != SAM3D_CACHE_SCHEMA:
                raise ValueError(f"Unsupported SAM3D cache schema: {schema}")
            frames = np.asarray(data["frame_indices"], dtype=np.int64)
            tids = tuple(str(x) for x in np.asarray(data["track_ids"]).tolist())
            boxes = np.asarray(data["boxes_xyxy"], dtype=np.float32)
            s2d = np.asarray(data["skel_2d_px"], dtype=np.float32)
            s3d = np.asarray(data["skel_3d_relative_m"], dtype=np.float32)
            valid = np.asarray(data["valid_mask"], dtype=bool)
            names = tuple(str(x) for x in np.asarray(data["joint_names"]).tolist())
            semantic = bool(np.asarray(data["semantic_mapping_validated"]).item())
            meta_raw = str(np.asarray(data["metadata_json"]).item())
            metadata = json.loads(meta_raw) if meta_raw else {}
        obj = cls(p, frames, tids, boxes, s2d, s3d, valid, names, semantic, metadata)
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.frame_indices.ndim != 1:
            raise ValueError("frame_indices must be 1D")
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("track_ids must be unique")
        if self.boxes_xyxy.shape != (self.T, self.N, 4):
            raise ValueError(f"boxes_xyxy shape mismatch: {self.boxes_xyxy.shape}")
        if self.skel_2d_px.shape != (self.T, self.N, FC_EXPECTED_JOINTS, 2):
            raise ValueError(f"skel_2d_px shape mismatch: {self.skel_2d_px.shape}")
        if self.skel_3d_relative_m.shape != (self.T, self.N, FC_EXPECTED_JOINTS, 3):
            raise ValueError(f"skel_3d_relative_m shape mismatch: {self.skel_3d_relative_m.shape}")
        if self.valid_mask.shape != (self.T, self.N):
            raise ValueError(f"valid_mask shape mismatch: {self.valid_mask.shape}")
        if len(self.joint_names) != FC_EXPECTED_JOINTS:
            raise ValueError(f"Expected {FC_EXPECTED_JOINTS} joint names, got {len(self.joint_names)}")
        if np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")


def save_sam3d_cache(
    path: str | Path,
    *,
    frame_indices: Sequence[int],
    track_ids: Sequence[str],
    boxes_xyxy: np.ndarray,
    skel_2d_px: np.ndarray,
    skel_3d_relative_m: np.ndarray,
    valid_mask: np.ndarray,
    joint_names: Sequence[str] | None = None,
    semantic_mapping_validated: bool = False,
    metadata: dict | None = None,
) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(joint_names or [f"sam3d_{i:02d}" for i in range(FC_EXPECTED_JOINTS)])
    payload = {
        "schema_version": np.array(SAM3D_CACHE_SCHEMA),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "track_ids": np.asarray([str(x) for x in track_ids]),
        "boxes_xyxy": np.asarray(boxes_xyxy, dtype=np.float32),
        "skel_2d_px": np.asarray(skel_2d_px, dtype=np.float32),
        "skel_3d_relative_m": np.asarray(skel_3d_relative_m, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "joint_names": np.asarray(names),
        "semantic_mapping_validated": np.asarray(bool(semantic_mapping_validated)),
        "metadata_json": np.array(json.dumps(metadata or {}, ensure_ascii=True)),
    }
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(p)
    Sam3DCache.load(p)
    return p
