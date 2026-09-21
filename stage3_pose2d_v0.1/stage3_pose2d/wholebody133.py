from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
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
WHOLEBODY_KEYPOINT_NAMES: Tuple[str, ...] = (
    COCO_BODY_NAMES
    + COCO_FOOT_NAMES
    + tuple(f"face_{i:02d}" for i in range(68))
    + tuple(f"left_hand_{i:02d}" for i in range(21))
    + tuple(f"right_hand_{i:02d}" for i in range(21))
)
assert len(WHOLEBODY_KEYPOINT_NAMES) == 133

NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(WHOLEBODY_KEYPOINT_NAMES)}
BODY17 = tuple(range(17))
FEET6 = tuple(range(17, 23))
BODY23 = tuple(range(23))
FACE68 = tuple(range(23, 91))
LEFT_HAND21 = tuple(range(91, 112))
RIGHT_HAND21 = tuple(range(112, 133))

CORE_OFFSIDE_ANATOMY = tuple(
    NAME_TO_INDEX[n]
    for n in (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel",
    )
)

BODY_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
)

SYMMETRIC_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 20), (18, 21), (19, 22),
)

H36M17_NAMES: Tuple[str, ...] = (
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck_nose", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)


def validate_keypoint_names(names: Sequence[str]) -> None:
    if len(names) != 133:
        raise ValueError(f"Expected 133 keypoint names, got {len(names)}")
    if tuple(names) != WHOLEBODY_KEYPOINT_NAMES:
        for i, (actual, expected) in enumerate(zip(names, WHOLEBODY_KEYPOINT_NAMES)):
            if actual != expected:
                raise ValueError(f"WholeBody ordering mismatch at index {i}: {actual!r} != {expected!r}")
        raise ValueError("WholeBody keypoint ordering mismatch")


def keypoint_records_to_arrays(records: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    if len(records) != 133:
        raise ValueError(f"Expected 133 keypoints, got {len(records)}")
    xy = np.full((133, 2), np.nan, dtype=np.float32)
    scores = np.zeros(133, dtype=np.float32)
    names: List[str] = []
    for expected_idx, rec in enumerate(records):
        idx = int(rec.get("index", expected_idx))
        if idx != expected_idx:
            raise ValueError(f"Keypoint index mismatch: list position {expected_idx}, record index {idx}")
        name = str(rec.get("name", WHOLEBODY_KEYPOINT_NAMES[idx]))
        names.append(name)
        x, y = rec.get("x"), rec.get("y")
        if x is not None and y is not None:
            try:
                xf, yf = float(x), float(y)
                if np.isfinite(xf) and np.isfinite(yf):
                    xy[idx] = (xf, yf)
            except (TypeError, ValueError):
                pass
        try:
            scores[idx] = float(rec.get("raw_score", rec.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            scores[idx] = 0.0
    validate_keypoint_names(names)
    return xy, scores


def _mean_points(xy: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    pts = np.asarray([xy[i] for i in indices], dtype=np.float32)
    valid = np.isfinite(pts).all(axis=1)
    if not np.any(valid):
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return np.mean(pts[valid], axis=0)


def coco133_to_h36m17(xy: np.ndarray) -> np.ndarray:
    """Derive a standard H36M-17 view without changing the canonical 133-point source."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape != (133, 2):
        raise ValueError(f"Expected (133,2), got {xy.shape}")
    out = np.full((17, 2), np.nan, dtype=np.float32)
    lhip, rhip = NAME_TO_INDEX["left_hip"], NAME_TO_INDEX["right_hip"]
    lsho, rsho = NAME_TO_INDEX["left_shoulder"], NAME_TO_INDEX["right_shoulder"]
    pelvis = _mean_points(xy, (lhip, rhip))
    thorax = _mean_points(xy, (lsho, rsho))
    out[0] = pelvis
    out[1] = xy[rhip]
    out[2] = xy[NAME_TO_INDEX["right_knee"]]
    out[3] = xy[NAME_TO_INDEX["right_ankle"]]
    out[4] = xy[lhip]
    out[5] = xy[NAME_TO_INDEX["left_knee"]]
    out[6] = xy[NAME_TO_INDEX["left_ankle"]]
    out[7] = (pelvis + thorax) * 0.5 if np.isfinite(pelvis).all() and np.isfinite(thorax).all() else np.nan
    out[8] = thorax
    out[9] = xy[NAME_TO_INDEX["nose"]]
    head_candidates = [NAME_TO_INDEX[n] for n in ("left_eye", "right_eye", "left_ear", "right_ear", "nose")]
    out[10] = _mean_points(xy, head_candidates)
    out[11] = xy[lsho]
    out[12] = xy[NAME_TO_INDEX["left_elbow"]]
    out[13] = xy[NAME_TO_INDEX["left_wrist"]]
    out[14] = xy[rsho]
    out[15] = xy[NAME_TO_INDEX["right_elbow"]]
    out[16] = xy[NAME_TO_INDEX["right_wrist"]]
    return out
