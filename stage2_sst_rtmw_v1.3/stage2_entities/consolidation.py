from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import math

HUMAN_LABEL_IDS = frozenset({2, 3, 4, 5, 6})
CANDIDATE_LABEL_IDS = frozenset({2, 3})
BALL_LABEL_ID = 1

ROLE_BY_LABEL_ID = {
    2: "player",
    3: "goalkeeper",
    4: "referee",
    5: "referee",
    6: "other",
}
ROLE_ORDER = {"player": 0, "goalkeeper": 1, "referee": 2, "other": 3}
SUPERCLASS_BY_ROLE = {
    "player": "footballer",
    "goalkeeper": "footballer",
    "referee": "referee",
    "other": "other",
}


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _height(box: Sequence[float]) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def normalized_center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Center distance normalized by the smaller human-box height.

    Human boxes of the same physical person can differ materially across mutually
    exclusive SST classes. Normalizing by height is more stable than image diagonal
    for this local duplicate test, while remaining conservative for adjacent players.
    """
    ac = _center(a)
    bc = _center(b)
    dist = math.hypot(ac[0] - bc[0], ac[1] - bc[1])
    return dist / max(1.0, min(_height(a), _height(b)))


def area_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    aa = _area(a)
    bb = _area(b)
    if aa <= 0 or bb <= 0:
        return 0.0
    return min(aa, bb) / max(aa, bb)


def intersection_over_min_area(a: Sequence[float], b: Sequence[float]) -> float:
    aa = _area(a)
    bb = _area(b)
    denom = min(aa, bb)
    return _intersection_area(a, b) / denom if denom > 0 else 0.0


def should_merge_cross_class(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    high_iou_threshold: float = 0.82,
    low_iou_threshold: float = 0.60,
    max_center_distance: float = 0.18,
    min_area_similarity: float = 0.65,
    min_intersection_over_min: float = 0.75,
) -> Tuple[bool, str, Dict[str, float]]:
    """Return whether two *cross-class* boxes plausibly represent one physical human.

    M1 uses a two-path rule:
      1) very high IoU -> merge directly;
      2) moderate IoU -> merge only when centers, scale and containment agree.

    The second path addresses Player/GK and Player/Referee duplicate boxes observed in
    SoccerNet failure cases without globally lowering one IoU threshold for crowded scenes.
    """
    if int(a.get("label_id", -1)) == int(b.get("label_id", -1)):
        return False, "same_class", {}
    box_a = a["bbox_xyxy"]
    box_b = b["bbox_xyxy"]
    iou = box_iou(box_a, box_b)
    center = normalized_center_distance(box_a, box_b)
    area_sim = area_similarity(box_a, box_b)
    iom = intersection_over_min_area(box_a, box_b)
    stats = {
        "iou": float(iou),
        "normalized_center_distance": float(center),
        "area_similarity": float(area_sim),
        "intersection_over_min_area": float(iom),
    }
    if iou >= high_iou_threshold:
        return True, "high_iou", stats
    if (
        iou >= low_iou_threshold
        and center <= max_center_distance
        and area_sim >= min_area_similarity
        and iom >= min_intersection_over_min
    ):
        return True, "geometry_consistent", stats
    return False, "not_same_physical_human", stats


@dataclass
class PhysicalHumanHypothesis:
    physical_human_id: str
    representative_detection_id: str
    representative_label_id: int
    representative_label: str
    bbox_xyxy: List[float]
    detector_score: float
    source_detection_ids: List[str]
    class_evidence: Dict[str, float]
    role_evidence: Dict[str, float]
    superclass: str
    superclass_score: float
    superclass_margin: float
    superclass_status: str
    resolved_role: str
    role_score: float
    role_margin: float
    role_status: str
    pose_required: bool
    candidate_hint: bool
    consolidation_reason: str
    consolidation_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def consolidate_human_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    cross_class_iou_threshold: float | None = None,
    high_iou_threshold: float = 0.82,
    low_iou_threshold: float = 0.60,
    max_center_distance: float = 0.18,
    min_area_similarity: float = 0.65,
    min_intersection_over_min: float = 0.75,
    ambiguous_margin: float = 0.05,
) -> List[PhysicalHumanHypothesis]:
    """Collapse mutually-exclusive SST human classes into physical people.

    ``cross_class_iou_threshold`` is retained as a backwards-compatible alias for the
    direct/high-IoU rule. New code should prefer the explicit M1 geometry parameters.

    The output also exposes a hierarchical semantic view. ``superclass=footballer``
    groups Player+Goalkeeper because that distinction is not a Stage-3 eligibility
    boundary; referee/other remain non-footballer classes.
    """
    if cross_class_iou_threshold is not None:
        high_iou_threshold = float(cross_class_iou_threshold)
    for name, value in (
        ("high_iou_threshold", high_iou_threshold),
        ("low_iou_threshold", low_iou_threshold),
        ("max_center_distance", max_center_distance),
        ("min_area_similarity", min_area_similarity),
        ("min_intersection_over_min", min_intersection_over_min),
    ):
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if low_iou_threshold > high_iou_threshold:
        raise ValueError("low_iou_threshold must be <= high_iou_threshold")

    humans = [dict(d) for d in detections if int(d.get("label_id", -1)) in HUMAN_LABEL_IDS]
    n = len(humans)
    parent = list(range(n))
    merge_edges: List[Dict[str, Any]] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            merged, reason, stats = should_merge_cross_class(
                humans[i], humans[j],
                high_iou_threshold=high_iou_threshold,
                low_iou_threshold=low_iou_threshold,
                max_center_distance=max_center_distance,
                min_area_similarity=min_area_similarity,
                min_intersection_over_min=min_intersection_over_min,
            )
            if merged:
                union(i, j)
                merge_edges.append({
                    "a": str(humans[i].get("detection_id")),
                    "b": str(humans[j].get("detection_id")),
                    "reason": reason,
                    **stats,
                })

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for idx, item in enumerate(humans):
        groups.setdefault(find(idx), []).append(item)

    raw_groups: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for members in groups.values():
        representative = max(
            members,
            key=lambda d: (float(d.get("score", 0.0)), -int(d.get("label_id", 99))),
        )
        raw_groups.append((members, representative))
    raw_groups.sort(key=lambda pair: (_center(pair[1]["bbox_xyxy"])[0], _center(pair[1]["bbox_xyxy"])[1]))

    result: List[PhysicalHumanHypothesis] = []
    for number, (members, representative) in enumerate(raw_groups, start=1):
        class_evidence: Dict[str, float] = {}
        role_evidence = {role: 0.0 for role in ("player", "goalkeeper", "referee", "other")}
        member_ids = {str(x.get("detection_id")) for x in members}
        relevant_edges = [e for e in merge_edges if e["a"] in member_ids and e["b"] in member_ids]
        for item in members:
            label = str(item.get("label", item.get("label_id")))
            score = float(item.get("score", 0.0))
            class_evidence[label] = max(class_evidence.get(label, 0.0), score)
            role = ROLE_BY_LABEL_ID[int(item["label_id"])]
            role_evidence[role] = max(role_evidence[role], score)

        ranked = sorted(role_evidence.items(), key=lambda kv: (-kv[1], ROLE_ORDER[kv[0]]))
        role, role_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = float(role_score - second_score)
        role_status = "VALID" if margin >= ambiguous_margin else "DEGRADED"

        footballer_score = max(role_evidence["player"], role_evidence["goalkeeper"])
        referee_score = role_evidence["referee"]
        other_score = role_evidence["other"]
        super_scores = {
            "footballer": footballer_score,
            "referee": referee_score,
            "other": other_score,
        }
        super_ranked = sorted(super_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        superclass, superclass_score = super_ranked[0]
        superclass_second = super_ranked[1][1] if len(super_ranked) > 1 else 0.0
        superclass_margin = float(superclass_score - superclass_second)
        superclass_status = "VALID" if superclass_margin >= ambiguous_margin else "DEGRADED"

        excluded_score = max(referee_score, other_score)
        pose_required = bool(
            superclass == "footballer"
            or (footballer_score > 0 and footballer_score >= excluded_score - ambiguous_margin)
        )
        candidate_hint = superclass == "footballer"

        if len(members) == 1:
            reason = "single_detection"
        else:
            reasons = sorted({str(e.get("reason")) for e in relevant_edges})
            reason = "+".join(reasons) if reasons else "cross_class_group"

        result.append(PhysicalHumanHypothesis(
            physical_human_id=f"human_{number:03d}",
            representative_detection_id=str(representative["detection_id"]),
            representative_label_id=int(representative["label_id"]),
            representative_label=str(representative["label"]),
            bbox_xyxy=[float(x) for x in representative["bbox_xyxy"]],
            detector_score=float(representative["score"]),
            source_detection_ids=[str(x["detection_id"]) for x in sorted(members, key=lambda d: str(d["detection_id"]))],
            class_evidence={k: float(v) for k, v in sorted(class_evidence.items())},
            role_evidence={k: float(v) for k, v in role_evidence.items()},
            superclass=superclass,
            superclass_score=float(superclass_score),
            superclass_margin=superclass_margin,
            superclass_status=superclass_status,
            resolved_role=role,
            role_score=float(role_score),
            role_margin=margin,
            role_status=role_status,
            pose_required=pose_required,
            candidate_hint=candidate_hint,
            consolidation_reason=reason,
            consolidation_diagnostics={
                "member_count": len(members),
                "merge_edges": relevant_edges,
                "thresholds": {
                    "high_iou": float(high_iou_threshold),
                    "low_iou": float(low_iou_threshold),
                    "max_center_distance": float(max_center_distance),
                    "min_area_similarity": float(min_area_similarity),
                    "min_intersection_over_min": float(min_intersection_over_min),
                },
            },
        ))
    return result
