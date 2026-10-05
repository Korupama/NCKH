"""Stage-3 crop geometry and ownership diagnostics.

This module only inspects the Stage-2 boxes and Stage-3 pose coordinates.  It
does not detect people, alter keypoints, or write anything into Stage 2.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .rtmw_onnx import compute_crop_geometry
from .wholebody133 import BODY17


def _box_xyxy(value: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy box: {value!r}")
    return x1, y1, x2, y2


def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = _box_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _box_xyxy(a)
    bx1, by1, bx2, by2 = _box_xyxy(b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _intersection(a, b)
    return float(inter / max(1e-6, _area(a) + _area(b) - inter))


def _inside_fraction(points: np.ndarray, box: Sequence[float]) -> float:
    if points.size == 0:
        return 0.0
    x1, y1, x2, y2 = _box_xyxy(box)
    inside = (
        (points[:, 0] >= x1)
        & (points[:, 0] <= x2)
        & (points[:, 1] >= y1)
        & (points[:, 1] <= y2)
    )
    return float(np.mean(inside))


def analyze_crop(
    source_bbox_xyxy: Sequence[float],
    image_size_wh: Sequence[int | float] | None,
    *,
    expanded_bbox_xyxy: Sequence[float] | None = None,
    neighbors: Iterable[Mapping[str, Any]] = (),
    pose_xy: np.ndarray | None = None,
    pose_scores: np.ndarray | None = None,
    bbox_padding: float = 1.25,
    crop_scale: float = 1.0,
    input_size_wh: Sequence[int | float] = (288, 384),
    small_height_px: float = 24.0,
    small_area_px: float = 512.0,
    high_overlap_iou: float = 0.30,
    cross_person_fraction: float = 0.45,
) -> dict[str, Any]:
    """Return auditable crop/ownership evidence for one Stage-3 observation.

    ``visible_person_area_px`` is deliberately labelled as a bbox proxy.  A
    true silhouette/segmentation area is not available in the Stage-2 handoff,
    so this function never presents the proxy as a segmentation measurement.
    """

    source = _box_xyxy(source_bbox_xyxy)
    expanded = _box_xyxy(expanded_bbox_xyxy or source)
    width = height = None
    if image_size_wh is not None and len(image_size_wh) >= 2:
        width, height = float(image_size_wh[0]), float(image_size_wh[1])
        if width <= 0 or height <= 0:
            width = height = None

    geometry = compute_crop_geometry(
        expanded,
        input_width=int(input_size_wh[0]),
        input_height=int(input_size_wh[1]),
        bbox_padding=float(bbox_padding),
        crop_scale=float(crop_scale),
    )
    crop_rect = geometry["crop_rect_xyxy"]
    crop_center = np.asarray(geometry["crop_center_xy"], dtype=float)
    source_center = np.asarray([(source[0] + source[2]) * 0.5, (source[1] + source[3]) * 0.5], dtype=float)
    source_width, source_height = source[2] - source[0], source[3] - source[1]
    crop_width, crop_height = crop_rect[2] - crop_rect[0], crop_rect[3] - crop_rect[1]

    if width is None or height is None:
        border_fraction = None
        visible_bbox_area = None
        crop_visible_fraction = None
    else:
        crop_area = max(1e-6, crop_width * crop_height)
        crop_inside = max(0.0, min(crop_rect[2], width) - max(crop_rect[0], 0.0)) * max(0.0, min(crop_rect[3], height) - max(crop_rect[1], 0.0))
        border_fraction = float(1.0 - crop_inside / crop_area)
        visible_bbox_area = float(max(0.0, min(source[2], width) - max(source[0], 0.0)) * max(0.0, min(source[3], height) - max(source[1], 0.0)))
        crop_visible_fraction = float(crop_inside / crop_area)

    neighbor_records: list[dict[str, Any]] = []
    max_crop_iou = 0.0
    max_source_iou = 0.0
    max_crop_neighbor: str | None = None
    max_source_neighbor: str | None = None
    point_array = np.asarray(pose_xy, dtype=float) if pose_xy is not None else np.empty((0, 2), dtype=float)
    score_array = np.asarray(pose_scores, dtype=float).reshape(-1) if pose_scores is not None else np.empty((0,), dtype=float)
    body_indices = np.asarray(BODY17, dtype=int)
    if point_array.ndim != 2 or point_array.shape[-1:] != (2,):
        point_array = np.empty((0, 2), dtype=float)
    for index, neighbor in enumerate(neighbors):
        neighbor_box = neighbor.get("bbox_xyxy")
        if not neighbor_box:
            continue
        try:
            neighbor_box = _box_xyxy(neighbor_box)
        except (TypeError, ValueError):
            continue
        neighbor_id = str(neighbor.get("track_id") or neighbor.get("id") or f"neighbor_{index:02d}")
        if neighbor_id == str(neighbor.get("current_track_id", "")):
            continue
        source_iou = _iou(source, neighbor_box)
        crop_iou = _iou(crop_rect, neighbor_box)
        previous_source_iou = max_source_iou
        previous_crop_iou = max_crop_iou
        max_source_iou = max(max_source_iou, source_iou)
        max_crop_iou = max(max_crop_iou, crop_iou)
        if source_iou > previous_source_iou and source_iou > 0.0:
            max_source_neighbor = neighbor_id
        if crop_iou > previous_crop_iou and crop_iou > 0.0:
            max_crop_neighbor = neighbor_id
        neighbor_fraction = None
        if point_array.shape[0] >= int(np.max(body_indices)) + 1:
            available = np.isfinite(point_array[body_indices]).all(axis=1)
            if score_array.shape[0] >= int(np.max(body_indices)) + 1:
                available &= np.isfinite(score_array[body_indices]) & (score_array[body_indices] > 0.0)
            neighbor_fraction = _inside_fraction(point_array[body_indices][available], neighbor_box) if np.any(available) else 0.0
        neighbor_records.append({
            "track_id": neighbor_id,
            "bbox_xyxy": [float(x) for x in neighbor_box],
            "source_bbox_iou": float(source_iou),
            "crop_rect_iou": float(crop_iou),
            "body17_fraction_inside_neighbor": neighbor_fraction,
        })

    own_body_fraction = None
    body_center_xy = None
    body_center_offset_px = None
    body_center_offset_normalized = None
    max_neighbor_body_fraction = 0.0
    max_neighbor_body_id = None
    if point_array.shape[0] >= int(np.max(body_indices)) + 1:
        available = np.isfinite(point_array[body_indices]).all(axis=1)
        if score_array.shape[0] >= int(np.max(body_indices)) + 1:
            available &= np.isfinite(score_array[body_indices]) & (score_array[body_indices] > 0.0)
        if np.any(available):
            body_points = point_array[body_indices][available]
            own_body_fraction = _inside_fraction(body_points, source)
            # Median is deliberately used here: one displaced wrist/ankle must
            # not make an otherwise correctly owned pose look like a different
            # person.  This is ownership evidence, not a coordinate rewrite.
            body_center = np.median(body_points, axis=0)
            body_center_xy = [float(x) for x in body_center]
            body_center_offset_px = float(np.linalg.norm(body_center - source_center))
            body_center_offset_normalized = float(
                body_center_offset_px / max(np.hypot(source_width, source_height), 1e-6)
            )
            for record in neighbor_records:
                fraction = record.get("body17_fraction_inside_neighbor")
                if fraction is not None and float(fraction) > max_neighbor_body_fraction:
                    max_neighbor_body_fraction = float(fraction)
                    max_neighbor_body_id = record["track_id"]

    status_reasons: list[str] = []
    if width is None or height is None:
        status_reasons.append("missing_image_dimensions")
    if source_height < float(small_height_px) or _area(source) < float(small_area_px):
        status_reasons.append("person_bbox_too_small")
    if border_fraction is not None and border_fraction >= 0.35:
        status_reasons.append("crop_border_truncation_high")
    if max_crop_iou >= float(high_overlap_iou):
        status_reasons.append("crop_overlaps_neighbor")
    if max_neighbor_body_fraction >= float(cross_person_fraction) and (
        own_body_fraction is None or max_neighbor_body_fraction >= own_body_fraction + 0.10
    ):
        status_reasons.append("body17_overlaps_neighbor_more_than_owner")

    if "body17_overlaps_neighbor_more_than_owner" in status_reasons:
        status = "LIKELY_CROSS_PERSON"
    elif "crop_border_truncation_high" in status_reasons:
        status = "SEVERE_BORDER_TRUNCATION"
    elif "crop_overlaps_neighbor" in status_reasons:
        status = "HIGH_OVERLAP"
    elif "person_bbox_too_small" in status_reasons:
        status = "TOO_SMALL"
    elif status_reasons:
        status = "UNCERTAIN"
    else:
        status = "OK"

    if max_neighbor_body_fraction >= float(cross_person_fraction) and (
        own_body_fraction is None or max_neighbor_body_fraction >= own_body_fraction + 0.10
    ):
        ownership_status = "NEIGHBOR_DOMINANT"
    elif own_body_fraction is None:
        ownership_status = "UNKNOWN"
    elif own_body_fraction < 0.45:
        ownership_status = "OUTSIDE_SOURCE"
    elif body_center_offset_normalized is not None and body_center_offset_normalized > 1.10:
        ownership_status = "CENTER_MISMATCH"
    elif own_body_fraction < 0.70 or (
        body_center_offset_normalized is not None and body_center_offset_normalized > 0.65
    ):
        ownership_status = "WEAK_SUPPORT"
    else:
        ownership_status = "SUPPORTED"

    return {
        "crop_status": status,
        "status_reasons": status_reasons,
        "source_bbox_xyxy": [float(x) for x in source],
        "expanded_pose_bbox_xyxy": [float(x) for x in expanded],
        "crop_center_xy": [float(x) for x in crop_center],
        "crop_scale_xy": [float(x) for x in geometry["crop_scale_xy"]],
        "crop_rect_xyxy": [float(x) for x in crop_rect],
        "affine_matrix_source_to_model": geometry["affine_matrix_source_to_model"],
        "model_input_size_wh": [int(input_size_wh[0]), int(input_size_wh[1])],
        "bbox_padding": float(bbox_padding),
        "crop_scale": float(crop_scale),
        "image_size_wh": None if width is None else [int(width), int(height)],
        "normalization": "RTMW mean/std in BGR order; cache does not preserve normalized tensor",
        "crop_geometry_source": "reconstructed_from_bbox_and_stage3_config; raw Stage-2 cache may not store affine",
        "source_bbox_area_px": float(_area(source)),
        "visible_person_area_px": visible_bbox_area,
        "visible_person_area_method": "clipped_stage2_bbox_area_proxy",
        "person_bbox_aspect_ratio": float(source_width / max(source_height, 1e-6)),
        "crop_border_truncation_fraction": border_fraction,
        "crop_visible_fraction": crop_visible_fraction,
        "crop_center_to_source_center_px": float(np.linalg.norm(crop_center - source_center)),
        "crop_center_to_source_center_normalized": float(np.linalg.norm(crop_center - source_center) / max(np.hypot(source_width, source_height), 1e-6)),
        "neighbor_max_source_bbox_iou": float(max_source_iou),
        "neighbor_max_source_bbox_track_id": max_source_neighbor,
        "neighbor_max_crop_iou": float(max_crop_iou),
        "neighbor_max_crop_track_id": max_crop_neighbor,
        "body17_fraction_inside_source_bbox": own_body_fraction,
        "body17_center_xy": body_center_xy,
        "body17_center_to_source_center_px": body_center_offset_px,
        "body17_center_to_source_center_normalized": body_center_offset_normalized,
        "ownership_status": ownership_status,
        "body17_fraction_inside_max_neighbor": float(max_neighbor_body_fraction),
        "body17_max_neighbor_track_id": max_neighbor_body_id,
        "neighbors": neighbor_records,
    }
