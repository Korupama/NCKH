from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ...wholebody import POSE23_NAMES
from .cache import MHR70_NAMES


@dataclass(frozen=True)
class JointMappingEntry:
    canonical_name: str
    rtmw_index: int
    sam3d_index: int
    sam3d_name: str


_ALIAS_TO_CANONICAL = {
    "left_big_toe_tip": "left_big_toe",
    "left_small_toe_tip": "left_small_toe",
    "right_big_toe_tip": "right_big_toe",
    "right_small_toe_tip": "right_small_toe",
}


def _sam_name_to_canonical(name: str) -> str:
    return _ALIAS_TO_CANONICAL.get(name, name)


def build_default_mapping(joint_names: Iterable[str] = MHR70_NAMES) -> tuple[JointMappingEntry, ...]:
    native = tuple(str(x) for x in joint_names)
    lookup: dict[str, int] = {}
    for i, name in enumerate(native):
        canonical = _sam_name_to_canonical(name)
        # MHR70 has unique canonical body/foot semantics after the toe-tip aliases.
        lookup.setdefault(canonical, i)
    entries = []
    for rtmw_idx, canonical_name in enumerate(POSE23_NAMES):
        if canonical_name not in lookup:
            raise ValueError(f"SAM3D MHR70 has no verified mapping for Stage-4 canonical joint {canonical_name}")
        sam_idx = lookup[canonical_name]
        entries.append(JointMappingEntry(canonical_name, rtmw_idx, sam_idx, native[sam_idx]))
    if len(entries) != 23:
        raise AssertionError("Expected complete 23-joint canonical mapping")
    return tuple(entries)


DEFAULT_MAPPING = build_default_mapping()
MAPPING_BY_CANONICAL = {x.canonical_name: x for x in DEFAULT_MAPPING}
GROUND_PRIMARY = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
GROUND_FALLBACK = ("left_ankle", "right_ankle")


def mapping_provenance() -> dict:
    return {
        "schema": "sam3d-mhr70-to-stage4-pose23-1.0",
        "source_schema": "SAM 3D Body MHR70 official ordering",
        "target_schema": "Stage4 canonical Pose23 / RTMW WholeBody first 23",
        "entries": [
            {
                "canonical_name": e.canonical_name,
                "rtmw_index": e.rtmw_index,
                "sam3d_index": e.sam3d_index,
                "sam3d_name": e.sam3d_name,
            }
            for e in DEFAULT_MAPPING
        ],
    }
