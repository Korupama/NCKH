from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np

COCO_BODY_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_FOOT_NAMES: Tuple[str, ...] = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
WHOLEBODY_NAMES: Tuple[str, ...] = (
    COCO_BODY_NAMES
    + COCO_FOOT_NAMES
    + tuple(f"face_{i:02d}" for i in range(68))
    + tuple(f"left_hand_{i:02d}" for i in range(21))
    + tuple(f"right_hand_{i:02d}" for i in range(21))
)
assert len(WHOLEBODY_NAMES) == 133
POSE23_NAMES: Tuple[str, ...] = WHOLEBODY_NAMES[:23]
NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(WHOLEBODY_NAMES)}

# Project-scope mask for downstream legal-body geometry. Shoulder is retained;
# the arm branch below the shoulder is excluded. Stage 5 remains responsible
# for constructing the actual legal-body surface/extent.
ARM_EXCLUDED = frozenset({"left_elbow", "right_elbow", "left_wrist", "right_wrist"})
LEGAL_GEOMETRY_CANDIDATE_23 = tuple(name not in ARM_EXCLUDED for name in POSE23_NAMES)


@dataclass(frozen=True)
class BonePrior:
    name: str
    a: int
    b: int
    length_ratio_to_height: float
    sigma_ratio_to_height: float


# Deliberately broad anthropometric priors. These stabilize monocular metric
# scale; they are not treated as player-specific ground truth.
BONE_PRIORS: Tuple[BonePrior, ...] = (
    BonePrior("shoulder_width", 5, 6, 0.235, 0.055),
    BonePrior("left_upper_arm", 5, 7, 0.186, 0.045),
    BonePrior("right_upper_arm", 6, 8, 0.186, 0.045),
    BonePrior("left_forearm", 7, 9, 0.146, 0.040),
    BonePrior("right_forearm", 8, 10, 0.146, 0.040),
    BonePrior("left_torso", 5, 11, 0.305, 0.060),
    BonePrior("right_torso", 6, 12, 0.305, 0.060),
    BonePrior("hip_width", 11, 12, 0.190, 0.050),
    BonePrior("left_thigh", 11, 13, 0.245, 0.045),
    BonePrior("right_thigh", 12, 14, 0.245, 0.045),
    BonePrior("left_shank", 13, 15, 0.246, 0.045),
    BonePrior("right_shank", 14, 16, 0.246, 0.045),
    BonePrior("left_ankle_bigtoe", 15, 17, 0.150, 0.055),
    BonePrior("left_ankle_smalltoe", 15, 18, 0.145, 0.055),
    BonePrior("left_ankle_heel", 15, 19, 0.075, 0.040),
    BonePrior("right_ankle_bigtoe", 16, 20, 0.150, 0.055),
    BonePrior("right_ankle_smalltoe", 16, 21, 0.145, 0.055),
    BonePrior("right_ankle_heel", 16, 22, 0.075, 0.040),
)

# Initial plane-height fractions used only for numerical initialization. They
# are not retained as hard pose assumptions after optimization.
NOMINAL_Z_FRACTION = {
    "nose": 0.93,
    "left_eye": 0.95, "right_eye": 0.95,
    "left_ear": 0.94, "right_ear": 0.94,
    "left_shoulder": 0.82, "right_shoulder": 0.82,
    "left_elbow": 0.66, "right_elbow": 0.66,
    "left_wrist": 0.53, "right_wrist": 0.53,
    "left_hip": 0.53, "right_hip": 0.53,
    "left_knee": 0.285, "right_knee": 0.285,
    "left_ankle": 0.055, "right_ankle": 0.055,
    "left_big_toe": 0.015, "left_small_toe": 0.015, "left_heel": 0.018,
    "right_big_toe": 0.015, "right_small_toe": 0.015, "right_heel": 0.018,
}

LEFT_FOOT = (15, 17, 18, 19)
RIGHT_FOOT = (16, 20, 21, 22)
LEFT_GROUND_SURFACE = (17, 18, 19)
RIGHT_GROUND_SURFACE = (20, 21, 22)


def derive_pose_features(xyz23: np.ndarray) -> Dict[str, object]:
    xyz = np.asarray(xyz23, dtype=np.float64)
    if xyz.shape != (23, 3):
        raise ValueError(f"Expected (23,3), got {xyz.shape}")

    def mean_valid(indices: Iterable[int]):
        pts = xyz[list(indices)]
        valid = np.isfinite(pts).all(axis=1)
        if not np.any(valid):
            return None
        return np.mean(pts[valid], axis=0).tolist()

    left_distal = mean_valid((17, 18))
    right_distal = mean_valid((20, 21))

    def foot_proxy(ankle: int, heel: int, distal):
        parts: List[np.ndarray] = []
        if np.isfinite(xyz[ankle]).all():
            parts.append(xyz[ankle])
        if np.isfinite(xyz[heel]).all():
            parts.append(xyz[heel])
        if distal is not None:
            parts.append(np.asarray(distal, dtype=np.float64))
        if not parts:
            return None
        return np.mean(np.stack(parts), axis=0).tolist()

    return {
        "head_center_xyz_m": mean_valid((0, 1, 2, 3, 4)),
        "shoulder_center_xyz_m": mean_valid((5, 6)),
        "pelvis_center_xyz_m": mean_valid((11, 12)),
        "left_distal_foot_xyz_m": left_distal,
        "right_distal_foot_xyz_m": right_distal,
        "left_foot_proxy_xyz_m": foot_proxy(15, 19, left_distal),
        "right_foot_proxy_xyz_m": foot_proxy(16, 22, right_distal),
    }
