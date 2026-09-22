"""Phase-2 contract for task-aligned football Pose23 annotations.

This module defines annotation validation and the prediction-only view used to
compare Stage-3 WholeBody133 output with Pose23 ground truth.  It intentionally
does not assign legal/offside semantics or create World-State coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import json
import math

from .wholebody133 import WHOLEBODY_KEYPOINT_NAMES


POSE23_SCHEMA_VERSION = "pose23-at-t0-1.0"
POSE23_MANIFEST_SCHEMA_VERSION = "pose23-t0-manifest-1.0"
POSE23_NAMES = tuple(WHOLEBODY_KEYPOINT_NAMES[:23])
POSE23_COORDINATE_SPACE = "RAW_DISTORTED_PIXEL"
POSE23_SPLITS = frozenset({"train", "validation", "test"})
POSE23_VISIBILITY = frozenset({
    "VISIBLE",
    "OCCLUDED",
    "TRUNCATED",
    "OUT_OF_FRAME",
    "NOT_ANNOTATED",
})
POSE23_REVIEW_STATUS = frozenset({
    "SINGLE_ANNOTATOR",
    "DOUBLE_REVIEWED",
    "ADJUDICATED",
})
POSE23_SCALE_BINS = frozenset({"SMALL", "MEDIUM", "LARGE", "UNKNOWN"})
POSE23_OCCLUSION_LEVELS = frozenset({"NONE", "PARTIAL", "HEAVY", "UNKNOWN"})
POSE23_BLUR_LEVELS = frozenset({"NONE", "MILD", "SEVERE", "UNKNOWN"})
POSE23_VIEWS = frozenset({"FRONT", "BACK", "SIDE", "THREE_QUARTER", "UNKNOWN"})
POSE23_CROWD_LEVELS = frozenset({"ISOLATED", "MODERATE", "CROWDED", "UNKNOWN"})
POSE23_AUTHORIZATION_STATUS = frozenset({
    "UNSELECTED",
    "AUTHORIZED",
    "RESTRICTED_EXTERNAL",
    "NOT_AUTHORIZED",
})

_COORDINATE_REQUIRED_VISIBILITY = frozenset({"VISIBLE", "OCCLUDED", "TRUNCATED"})


class Pose23ValidationError(ValueError):
    """Raised when a Pose23 sample or manifest violates the frozen contract."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("Pose23 validation failed:\n" + "\n".join(f"- {x}" for x in self.issues))


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_bbox(bbox: Any, path: str, issues: List[str]) -> None:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        issues.append(f"{path}: expected [x1,y1,x2,y2]")
        return
    if not all(_finite_number(value) for value in bbox):
        issues.append(f"{path}: all coordinates must be finite numbers")
        return
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        issues.append(f"{path}: x2>x1 and y2>y1 are required")


def _validate_keypoint(record: Any, index: int, issues: List[str]) -> None:
    path = f"keypoints_23[{index}]"
    if not isinstance(record, Mapping):
        issues.append(f"{path}: expected an object")
        return
    if record.get("index") != index:
        issues.append(f"{path}.index: expected {index}, got {record.get('index')!r}")
    if record.get("name") != POSE23_NAMES[index]:
        issues.append(f"{path}.name: expected {POSE23_NAMES[index]!r}, got {record.get('name')!r}")

    visibility = record.get("visibility")
    if visibility not in POSE23_VISIBILITY:
        issues.append(f"{path}.visibility: unsupported value {visibility!r}")

    x, y = record.get("x"), record.get("y")
    has_xy = _finite_number(x) and _finite_number(y)
    if visibility in _COORDINATE_REQUIRED_VISIBILITY and not has_xy:
        issues.append(f"{path}: finite x/y required for visibility={visibility!r}")
    if visibility in {"OUT_OF_FRAME", "NOT_ANNOTATED"} and has_xy:
        issues.append(f"{path}: x/y must be null for visibility={visibility!r}")

    confidence = record.get("annotation_confidence")
    if not _finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
        issues.append(f"{path}.annotation_confidence: expected a number in [0,1]")
    if record.get("review_status") not in POSE23_REVIEW_STATUS:
        issues.append(f"{path}.review_status: unsupported value {record.get('review_status')!r}")


def validate_pose23_sample(sample: Mapping[str, Any], *, path: str = "sample") -> List[str]:
    """Return all contract issues for one inline Pose23 annotation sample."""

    issues: List[str] = []
    if not isinstance(sample, Mapping):
        return [f"{path}: expected an object"]

    for field in (
        "sample_id", "case_id", "video_id", "match_id", "sequence_id",
        "split_group_id", "track_id", "image_path",
    ):
        if not _non_empty_string(sample.get(field)):
            issues.append(f"{path}.{field}: required non-empty string")

    split = sample.get("split")
    if split not in POSE23_SPLITS:
        issues.append(f"{path}.split: expected one of {sorted(POSE23_SPLITS)}, got {split!r}")

    frame_index = sample.get("frame_index")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        issues.append(f"{path}.frame_index: expected a non-negative integer")

    if sample.get("coordinate_space") != POSE23_COORDINATE_SPACE:
        issues.append(f"{path}.coordinate_space: must be {POSE23_COORDINATE_SPACE!r}")
    if sample.get("bbox_coordinate_space") != POSE23_COORDINATE_SPACE:
        issues.append(f"{path}.bbox_coordinate_space: must be {POSE23_COORDINATE_SPACE!r}")
    if sample.get("t0_selection") != "USER_SELECTED":
        issues.append(f"{path}.t0_selection: must be 'USER_SELECTED'")
    _validate_bbox(sample.get("bbox_xyxy"), f"{path}.bbox_xyxy", issues)

    image_width, image_height = sample.get("image_width"), sample.get("image_height")
    for name, value in (("image_width", image_width), ("image_height", image_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            issues.append(f"{path}.{name}: expected a positive integer")

    keypoints = sample.get("keypoints_23")
    if not isinstance(keypoints, list) or len(keypoints) != 23:
        issues.append(f"{path}.keypoints_23: expected exactly 23 records")
    else:
        for index, record in enumerate(keypoints):
            _validate_keypoint(record, index, issues)
        if isinstance(image_width, int) and image_width > 0 and isinstance(image_height, int) and image_height > 0:
            for index, record in enumerate(keypoints):
                if not isinstance(record, Mapping):
                    continue
                if record.get("visibility") not in _COORDINATE_REQUIRED_VISIBILITY:
                    continue
                x, y = record.get("x"), record.get("y")
                if _finite_number(x) and _finite_number(y):
                    if not (0.0 <= float(x) < image_width and 0.0 <= float(y) < image_height):
                        issues.append(
                            f"{path}.keypoints_23[{index}]: annotated coordinate must lie inside image bounds"
                        )

    tags = sample.get("tags")
    if not isinstance(tags, Mapping):
        issues.append(f"{path}.tags: required object")
    else:
        enum_fields = (
            ("scale_bin", POSE23_SCALE_BINS),
            ("occlusion_level", POSE23_OCCLUSION_LEVELS),
            ("motion_blur", POSE23_BLUR_LEVELS),
            ("view", POSE23_VIEWS),
            ("crowd_level", POSE23_CROWD_LEVELS),
        )
        for name, allowed in enum_fields:
            if tags.get(name) not in allowed:
                issues.append(f"{path}.tags.{name}: unsupported value {tags.get(name)!r}")
        if not isinstance(tags.get("border_truncated"), bool):
            issues.append(f"{path}.tags.border_truncated: expected boolean")
        if not _finite_number(tags.get("player_pixel_height")) or float(tags["player_pixel_height"]) <= 0:
            issues.append(f"{path}.tags.player_pixel_height: expected a positive number")

    if sample.get("annotation_review_status") not in POSE23_REVIEW_STATUS:
        issues.append(
            f"{path}.annotation_review_status: unsupported value {sample.get('annotation_review_status')!r}"
        )
    return issues


def validate_pose23_manifest(manifest: Mapping[str, Any], *, allow_empty: bool = False) -> List[str]:
    """Return all issues for a Pose23 manifest, including split leakage checks."""

    issues: List[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest: expected an object"]
    if manifest.get("schema_version") != POSE23_MANIFEST_SCHEMA_VERSION:
        issues.append(f"manifest.schema_version: expected {POSE23_MANIFEST_SCHEMA_VERSION!r}")
    if manifest.get("coordinate_space") != POSE23_COORDINATE_SPACE:
        issues.append(f"manifest.coordinate_space: must be {POSE23_COORDINATE_SPACE!r}")
    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, Mapping):
        issues.append("manifest.source_provenance: required object")
    else:
        for field in ("source_name", "license_or_permission", "authorization_status"):
            if not _non_empty_string(provenance.get(field)):
                issues.append(f"manifest.source_provenance.{field}: required non-empty string")
        if provenance.get("authorization_status") not in POSE23_AUTHORIZATION_STATUS:
            issues.append(
                "manifest.source_provenance.authorization_status: unsupported value "
                f"{provenance.get('authorization_status')!r}"
            )
    schema = manifest.get("keypoint_schema")
    if not isinstance(schema, Mapping):
        issues.append("manifest.keypoint_schema: required object")
    else:
        if schema.get("name") != "POSE23_AT_T0":
            issues.append("manifest.keypoint_schema.name: expected 'POSE23_AT_T0'")
        if schema.get("count") != 23:
            issues.append("manifest.keypoint_schema.count: expected 23")
        if tuple(schema.get("names") or ()) != POSE23_NAMES:
            issues.append("manifest.keypoint_schema.names: canonical WholeBody first-23 ordering required")

    policy = manifest.get("evaluation_policy")
    if not isinstance(policy, Mapping):
        issues.append("manifest.evaluation_policy: required object")
    else:
        groups = [policy.get("primary"), policy.get("secondary"), policy.get("excluded")]
        if any(not isinstance(group, list) for group in groups):
            issues.append("manifest.evaluation_policy: primary/secondary/excluded must be lists")
        else:
            flattened = groups[0] + groups[1] + groups[2]
            if len(flattened) != len(set(flattened)) or set(flattened) != set(POSE23_VISIBILITY):
                issues.append("manifest.evaluation_policy: visibility values must form a disjoint complete partition")

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        issues.append("manifest.samples: expected a list")
        return issues
    if not samples and not allow_empty:
        issues.append("manifest.samples: empty manifest is only valid with allow_empty=True")

    sample_ids = set()
    split_groups: Dict[str, str] = {}
    sequence_groups: Dict[tuple[str, str], str] = {}
    sample_keys = set()
    for index, sample in enumerate(samples):
        sample_issues = validate_pose23_sample(sample, path=f"manifest.samples[{index}]")
        issues.extend(sample_issues)
        if not isinstance(sample, Mapping):
            continue
        sample_id = sample.get("sample_id")
        if sample_id in sample_ids:
            issues.append(f"manifest.samples[{index}].sample_id: duplicate {sample_id!r}")
        sample_ids.add(sample_id)
        split = sample.get("split")
        split_group = sample.get("split_group_id")
        if _non_empty_string(split_group) and split in POSE23_SPLITS:
            previous = split_groups.setdefault(split_group, split)
            if previous != split:
                issues.append(
                    f"split leakage: split_group_id={split_group!r} appears in both {previous!r} and {split!r}"
                )
        match_id, sequence_id = sample.get("match_id"), sample.get("sequence_id")
        if _non_empty_string(match_id) and _non_empty_string(sequence_id) and split in POSE23_SPLITS:
            key = (match_id, sequence_id)
            previous = sequence_groups.setdefault(key, split)
            if previous != split:
                issues.append(f"split leakage: match/sequence={key!r} appears in both {previous!r} and {split!r}")
        frame_key = (match_id, sequence_id, sample.get("frame_index"), sample.get("track_id"))
        if frame_key in sample_keys:
            issues.append(f"manifest.samples[{index}]: duplicate match/sequence/frame/track {frame_key!r}")
        sample_keys.add(frame_key)

    return issues


def load_and_validate_manifest(path: str | Path, *, allow_empty: bool = False) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    issues = validate_pose23_manifest(manifest, allow_empty=allow_empty)
    if issues:
        raise Pose23ValidationError(issues)
    return manifest


def wholebody133_to_pose23(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Stage-3 WholeBody133 prediction records to a 23-point view.

    This is prediction conversion only.  It preserves raw coordinates and
    Stage-3 evidence fields; it does not invent GT visibility labels.
    """

    if len(records) != 133:
        raise ValueError(f"Expected 133 WholeBody records, got {len(records)}")
    output: List[Dict[str, Any]] = []
    for index, expected_name in enumerate(POSE23_NAMES):
        record = records[index]
        if int(record.get("index", -1)) != index or record.get("name") != expected_name:
            raise ValueError(f"WholeBody133 ordering mismatch at index {index}")
        output.append({
            "index": index,
            "name": expected_name,
            "x": record.get("x"),
            "y": record.get("y"),
            "raw_model_score": record.get("raw_model_score"),
            "state": record.get("state", "MISSING"),
            "source": record.get("source"),
            "coordinate_evidence_kind": record.get("coordinate_evidence_kind"),
            "temporal_estimate_xy": record.get("temporal_estimate_xy"),
        })
    return output


def pose23_manifest_template() -> Dict[str, Any]:
    """Return a valid empty manifest template for annotation tooling."""

    return {
        "schema_version": POSE23_MANIFEST_SCHEMA_VERSION,
        "dataset_id": "football_pose23_at_t0",
        "dataset_version": "0.1.0",
        "task": "broadcast_football_pose23_at_user_selected_frame",
        "coordinate_space": POSE23_COORDINATE_SPACE,
        "t0_selection_policy": "USER_SELECTED",
        "source_provenance": {
            "source_name": "UNSELECTED",
            "license_or_permission": "UNSELECTED",
            "authorization_status": "UNSELECTED",
            "raw_media_policy": "keep_authorized_media_outside_repository",
        },
        "keypoint_schema": {
            "name": "POSE23_AT_T0",
            "count": 23,
            "names": list(POSE23_NAMES),
        },
        "split_policy": {
            "group_key": "split_group_id",
            "recommended_split_group": "match_id",
            "forbid_match_sequence_cross_split": True,
            "adjacent_frame_leakage_guard": "all samples from one match/sequence stay in one split",
            "test_locked": False,
        },
        "evaluation_policy": {
            "primary": ["VISIBLE"],
            "secondary": ["OCCLUDED"],
            "excluded": ["TRUNCATED", "OUT_OF_FRAME", "NOT_ANNOTATED"],
            "report_slices_separately": ["TRUNCATED", "OCCLUDED", "OUT_OF_FRAME"],
        },
        "annotation_policy_version": "pose23-annotation-guideline-1.0",
        "samples": [],
    }
