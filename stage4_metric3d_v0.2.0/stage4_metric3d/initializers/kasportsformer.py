from __future__ import annotations

"""KASportsFormer integration boundary.

The official KASportsFormer repository has its own environment and data layout.
Stage 4 therefore treats it as an optional external initializer worker, just like
RTMW3D. This module converts externally generated 17-joint relative 3D outputs
into the common cache schema without pretending that unavailable WholeBody foot
landmarks exist.

The v0.1 production path is RTMW3D-L. KASportsFormer is reserved for controlled
A/B evaluation after the primary metric optimizer is validated.
"""

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
import numpy as np

from .cache import write_jsonl

# H36M-style 17-joint output cannot faithfully populate all WholeBody133 points.
# We intentionally leave unobserved entries NaN; Stage-4 cache loader accepts
# them and the optimizer only consumes finite relative-depth evidence.

H36M17_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

# Exact COCO17 correspondences plus a conservative nose proxy for H36M head.
# Pelvis/spine/thorax/neck are synthetic H36M joints and therefore are not
# copied into a named COCO/WholeBody slot. None means deliberately unmapped.
H36M17_TO_WHOLEBODY133: Sequence[Optional[int]] = (
    None,
    12,
    14,
    16,
    11,
    13,
    15,
    None,
    None,
    None,
    0,
    5,
    7,
    9,
    6,
    8,
    10,
)


def write_kasportsformer_cache(
    output_path: str | Path,
    predictions: Iterable[Mapping[str, object]],
    *,
    source_description: str = "official KASportsFormer external worker",
) -> Path:
    records = []
    for rec in predictions:
        xyz_value = rec.get("relative_xyz_17", rec.get("xyz"))
        if xyz_value is None:
            raise ValueError("KASportsFormer record requires relative_xyz_17 or xyz")
        xyz17 = np.asarray(xyz_value, dtype=np.float64)
        if xyz17.shape != (17, 3):
            raise ValueError(f"Expected relative_xyz_17 shape (17,3), got {xyz17.shape}")
        raw = np.full((133, 3), np.nan, dtype=np.float64)
        indices = list(rec.get("wholebody_indices", H36M17_TO_WHOLEBODY133))
        if len(indices) != 17:
            raise ValueError("KASportsFormer conversion requires explicit wholebody_indices[17]")
        for src_idx, dst_idx in enumerate(indices):
            if dst_idx is None or int(dst_idx) < 0:
                continue
            if int(dst_idx) >= 133:
                raise ValueError(f"WholeBody destination index is outside [0,132]: {dst_idx}")
            raw[int(dst_idx)] = xyz17[src_idx]
        rel_z = raw[:, 2].copy()
        if np.isfinite(rel_z[[11, 12]]).any():
            rel_z -= float(np.nanmean(rel_z[[11, 12]]))
        records.append({
            "record_type": "observation",
            "track_id": str(rec["track_id"]),
            "frame_index": int(rec["frame_index"]),
            "backend": "KASportsFormer",
            "raw_keypoints_133": raw.tolist(),
            "keypoint_scores_133": np.where(np.isfinite(raw[:, 2]), 1.0, np.nan).tolist(),
            "relative_depth_133": rel_z.tolist(),
            "global_position_known": False,
            "xy_used_by_stage4": False,
        })
    return write_jsonl(output_path, {
        "schema_version": "stage4-relative3d-initializer-cache-1.0",
        "backend": "KASportsFormer",
        "source": source_description,
        "global_metric_position_trusted": False,
    }, records)
