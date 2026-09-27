from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import cv2
import numpy as np

from .contracts import BallFrameState


# Standard association-football pitch markings for the canonical 105 x 68 m field
# used by Stage 1. Values are in metres.
CENTER_CIRCLE_RADIUS_M = 9.15
PENALTY_AREA_DEPTH_M = 16.5
GOAL_AREA_DEPTH_M = 5.5
PENALTY_SPOT_DISTANCE_M = 11.0
GOAL_WIDTH_M = 7.32
CORNER_ARC_RADIUS_M = 1.0
# Only a visualization depth; the goal frame is outside the calibrated pitch plane.
GOAL_RENDER_DEPTH_M = 2.0
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def render_selected_frame(video_path: str | Path, selected: dict[str, Any], output_path: str | Path) -> Path:
    fi = int(selected["frame_index"])
    source = Path(video_path)
    if source.suffix.lower() in IMAGE_SUFFIXES:
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot decode image {video_path}")
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            raise RuntimeError(f"Cannot decode frame {fi}")
    cand = selected.get("candidate")
    if cand:
        x1, y1, x2, y2 = [int(round(v)) for v in cand["bbox_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        u, v = [int(round(z)) for z in cand["center_uv"]]
        cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(img, f"BALL t0={fi} {selected.get('status')}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    xyz = selected.get("center_xyz_world_m")
    if xyz:
        cv2.putText(img, f"XYZ=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})m", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def _sample_arc(center_x: float, center_y: float, radius: float, start_deg: float, end_deg: float, count: int = 48) -> list[tuple[float, float]]:
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), count)
    return [(center_x + radius * math.cos(a), center_y + radius * math.sin(a)) for a in angles]


def _render_metric_pitch(
    *,
    length_m: float,
    width_m: float,
    scale: float,
    padding_px: int,
    show_dimension_labels: bool,
) -> tuple[np.ndarray, Any]:
    """Render a complete top-down pitch and return ``(image, world_to_pixel)``.

    The pitch coordinate system is the Stage-1 world convention: X along the
    goal-to-goal direction and Y along the touchline-to-touchline direction.
    """
    field_w = int(round(length_m * scale))
    field_h = int(round(width_m * scale))
    W = field_w + 2 * padding_px
    H = field_h + 2 * padding_px

    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Dark-green outer canvas, slightly lighter playing surface.
    img[:] = (28, 78, 28)
    img[padding_px:padding_px + field_h, padding_px:padding_px + field_w] = (35, 100, 35)

    def pix(x: float, y: float) -> tuple[int, int]:
        px = padding_px + int(round((x + length_m / 2.0) * scale))
        py = padding_px + int(round((width_m / 2.0 - y) * scale))
        return px, py

    white = (245, 245, 245)
    line = max(1, int(round(scale * 0.18)))
    thin = max(1, line - 1)
    spot_r = max(2, int(round(scale * 0.22)))

    # Outer boundary and halfway line.
    cv2.rectangle(img, pix(-length_m / 2.0, width_m / 2.0), pix(length_m / 2.0, -width_m / 2.0), white, line, cv2.LINE_AA)
    cv2.line(img, pix(0.0, width_m / 2.0), pix(0.0, -width_m / 2.0), white, line, cv2.LINE_AA)

    # Centre circle and centre mark.
    cv2.circle(img, pix(0.0, 0.0), int(round(CENTER_CIRCLE_RADIUS_M * scale)), white, line, cv2.LINE_AA)
    cv2.circle(img, pix(0.0, 0.0), spot_r, white, -1, cv2.LINE_AA)

    goal_line_x = length_m / 2.0
    penalty_half_width = GOAL_WIDTH_M / 2.0 + PENALTY_AREA_DEPTH_M
    goal_area_half_width = GOAL_WIDTH_M / 2.0 + GOAL_AREA_DEPTH_M

    # Both penalty areas and goal areas.
    for side in (-1.0, 1.0):
        gx = side * goal_line_x
        penalty_inner_x = side * (goal_line_x - PENALTY_AREA_DEPTH_M)
        goal_area_inner_x = side * (goal_line_x - GOAL_AREA_DEPTH_M)

        # Penalty area: 16.5 m deep, 40.32 m wide.
        cv2.rectangle(
            img,
            pix(min(gx, penalty_inner_x), penalty_half_width),
            pix(max(gx, penalty_inner_x), -penalty_half_width),
            white,
            line,
            cv2.LINE_AA,
        )
        # Goal area: 5.5 m deep, 18.32 m wide.
        cv2.rectangle(
            img,
            pix(min(gx, goal_area_inner_x), goal_area_half_width),
            pix(max(gx, goal_area_inner_x), -goal_area_half_width),
            white,
            line,
            cv2.LINE_AA,
        )

        # Penalty spot and penalty arc (the part outside the penalty area).
        penalty_spot_x = side * (goal_line_x - PENALTY_SPOT_DISTANCE_M)
        cv2.circle(img, pix(penalty_spot_x, 0.0), spot_r, white, -1, cv2.LINE_AA)
        theta = math.degrees(math.acos((PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DISTANCE_M) / CENTER_CIRCLE_RADIUS_M))
        if side < 0:
            arc = _sample_arc(penalty_spot_x, 0.0, CENTER_CIRCLE_RADIUS_M, -theta, theta)
        else:
            arc = _sample_arc(penalty_spot_x, 0.0, CENTER_CIRCLE_RADIUS_M, 180.0 - theta, 180.0 + theta)
        cv2.polylines(img, [np.asarray([pix(x, y) for x, y in arc], dtype=np.int32)], False, white, line, cv2.LINE_AA)

        # Goal frame rendered outside the pitch for orientation only.
        goal_back_x = side * (goal_line_x + GOAL_RENDER_DEPTH_M)
        cv2.rectangle(
            img,
            pix(min(gx, goal_back_x), GOAL_WIDTH_M / 2.0),
            pix(max(gx, goal_back_x), -GOAL_WIDTH_M / 2.0),
            white,
            thin,
            cv2.LINE_AA,
        )

        if show_dimension_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.38, scale / 22.0)
            text_thickness = 1
            # Put labels inside the corresponding top horizontal edges so the
            # meaning remains obvious without obscuring the trajectory.
            label_165 = "16.5 m"
            label_55 = "5.5 m"
            if side < 0:
                p165 = pix(-goal_line_x + PENALTY_AREA_DEPTH_M * 0.45, penalty_half_width - 0.8)
                p55 = pix(-goal_line_x + GOAL_AREA_DEPTH_M * 0.30, goal_area_half_width - 0.8)
            else:
                p165 = pix(goal_line_x - PENALTY_AREA_DEPTH_M * 0.95, penalty_half_width - 0.8)
                p55 = pix(goal_line_x - GOAL_AREA_DEPTH_M * 0.95, goal_area_half_width - 0.8)
            cv2.putText(img, label_165, p165, font, font_scale, white, text_thickness, cv2.LINE_AA)
            cv2.putText(img, label_55, p55, font, font_scale, white, text_thickness, cv2.LINE_AA)

    # Corner arcs, all 1 m radius.
    corner_specs = [
        (-goal_line_x, width_m / 2.0, -90.0, 0.0),
        (-goal_line_x, -width_m / 2.0, 0.0, 90.0),
        (goal_line_x, width_m / 2.0, 180.0, 270.0),
        (goal_line_x, -width_m / 2.0, 90.0, 180.0),
    ]
    for cx, cy, a0, a1 in corner_specs:
        arc = _sample_arc(cx, cy, CORNER_ARC_RADIUS_M, a0, a1, count=20)
        cv2.polylines(img, [np.asarray([pix(x, y) for x, y in arc], dtype=np.int32)], False, white, thin, cv2.LINE_AA)

    return img, pix


def _extract_frame_xyz(frame: BallFrameState | Mapping[str, Any]) -> tuple[int, list[float] | None]:
    if isinstance(frame, Mapping):
        return int(frame["frame_index"]), frame.get("selected_center_xyz_world_m")
    return int(frame.frame_index), frame.selected_center_xyz_world_m


def render_minimap(
    frames: Sequence[BallFrameState | Mapping[str, Any]],
    selected_frame: int,
    output_path: str | Path,
    *,
    length_m: float = 105.0,
    width_m: float = 68.0,
    scale: float = 9.0,
    padding_px: int = 30,
    show_dimension_labels: bool = True,
) -> Path:
    """Render the ball trajectory over a regulation-pitch minimap.

    v0.2.1 visualization includes the 16.5 m penalty areas, 5.5 m goal areas,
    penalty spots/arcs, centre circle, corner arcs and small goal frames.
    """
    img, pix = _render_metric_pitch(
        length_m=length_m,
        width_m=width_m,
        scale=scale,
        padding_px=padding_px,
        show_dimension_labels=show_dimension_labels,
    )

    track_points: list[tuple[int, tuple[int, int], list[float]]] = []
    for frame in frames:
        fi, xyz = _extract_frame_xyz(frame)
        if xyz is None:
            continue
        x, y = float(xyz[0]), float(xyz[1])
        if not (-length_m / 2.0 - 3.0 <= x <= length_m / 2.0 + 3.0 and -width_m / 2.0 - 3.0 <= y <= width_m / 2.0 + 3.0):
            continue
        track_points.append((fi, pix(x, y), [float(v) for v in xyz]))

    # Trajectory polyline first, then per-frame observations.
    if len(track_points) >= 2:
        pts = np.asarray([p for _fi, p, _xyz in track_points], dtype=np.int32)
        cv2.polylines(img, [pts], False, (225, 225, 225), 2, cv2.LINE_AA)

    selected_xyz: list[float] | None = None
    for fi, p, xyz in track_points:
        if fi == selected_frame:
            selected_xyz = xyz
            # Black halo keeps the t0 marker visible over white pitch markings.
            cv2.circle(img, p, 8, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(img, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 2, (245, 245, 245), -1, cv2.LINE_AA)

    # Small direction markers at the first and last valid point.
    if track_points:
        _fi0, p0, _xyz0 = track_points[0]
        _fi1, p1, _xyz1 = track_points[-1]
        cv2.circle(img, p0, 4, (255, 200, 80), -1, cv2.LINE_AA)
        cv2.circle(img, p1, 4, (80, 200, 255), -1, cv2.LINE_AA)

    # Compact diagnostic banner in the outer top margin.
    banner_y = max(19, padding_px - 8)
    if selected_xyz is not None:
        text = f"Ball trajectory | t0={selected_frame} | X={selected_xyz[0]:.2f} m  Y={selected_xyz[1]:.2f} m  Z={selected_xyz[2]:.2f} m"
    else:
        text = f"Ball trajectory | t0={selected_frame} | no valid world point at t0"
    cv2.putText(img, text, (padding_px, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def render_minimap_from_state_json(
    state_json: str | Path,
    output_path: str | Path,
    *,
    scale: float = 9.0,
    padding_px: int = 30,
    show_dimension_labels: bool = True,
) -> Path:
    """Re-render a minimap from an existing Stage-6 state without rerunning YOLO."""
    import json

    state = json.loads(Path(state_json).read_text(encoding="utf-8"))
    t0 = int(state["replay_context"]["selected_frame"])
    pitch = state.get("diagnostics", {}).get("pitch_dimensions") or {}
    length_m = float(pitch.get("length_m", 105.0))
    width_m = float(pitch.get("width_m", 68.0))
    return render_minimap(
        state["frames"],
        t0,
        output_path,
        length_m=length_m,
        width_m=width_m,
        scale=scale,
        padding_px=padding_px,
        show_dimension_labels=show_dimension_labels,
    )
