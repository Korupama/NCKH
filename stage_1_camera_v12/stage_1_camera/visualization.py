from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .contracts import CameraState
from .pitch_model import pitch_polylines, pitch_segments


def _camera_depth(cam: CameraState, xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    xc = (cam.R_world_to_camera @ (pts - cam.camera_center_world_m).T).T
    return xc[:, 2]


def _clip_world_segment_to_near_plane(
    cam: CameraState,
    p0: np.ndarray,
    p1: np.ndarray,
    near_depth_m: float = 1e-3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a world-space segment against the camera near plane.

    Camera depth is affine along a 3D segment. Clipping *before* projection is
    essential: projecting a segment that crosses Zc=0 creates a projective
    singularity, which can appear as long spurious image lines and can also
    remove the genuinely visible part of a pitch marking.
    """
    a = np.asarray(p0, dtype=np.float64).reshape(3).copy()
    b = np.asarray(p1, dtype=np.float64).reshape(3).copy()
    da, db = _camera_depth(cam, np.stack([a, b]))
    eps = float(near_depth_m)

    if not np.isfinite(da) or not np.isfinite(db):
        return None
    if da <= eps and db <= eps:
        return None

    if da <= eps:
        denom = db - da
        if abs(denom) < 1e-12:
            return None
        t = (eps - da) / denom
        a = a + t * (b - a)
        da = eps

    if db <= eps:
        denom = db - da
        if abs(denom) < 1e-12:
            return None
        t = (eps - da) / denom
        b = a + t * (b - a)

    return a, b


def _clip_line_to_image(
    uv0: np.ndarray,
    uv1: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky clipping in floating-point image coordinates.

    We intentionally avoid relying on integer clipping because a point close to
    the camera near-plane can project to extremely large coordinates.
    """
    p0 = np.asarray(uv0, dtype=np.float64).reshape(2)
    p1 = np.asarray(uv1, dtype=np.float64).reshape(2)
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return None

    x0, y0 = p0
    dx, dy = p1 - p0
    xmin, xmax = 0.0, float(width - 1)
    ymin, ymax = 0.0, float(height - 1)

    u0, u1 = 0.0, 1.0
    for p, q in [
        (-dx, x0 - xmin),
        ( dx, xmax - x0),
        (-dy, y0 - ymin),
        ( dy, ymax - y0),
    ]:
        if abs(p) < 1e-15:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > u1:
                return None
            u0 = max(u0, r)
        else:
            if r < u0:
                return None
            u1 = min(u1, r)

    if u0 > u1:
        return None
    return p0 + u0 * (p1 - p0), p0 + u1 * (p1 - p0)


def _draw_projected_polyline(
    canvas: np.ndarray,
    cam: CameraState,
    xyz: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    near_depth_m: float = 1e-3,
) -> None:
    """Draw a 3D polyline with camera-space and image-space clipping.

    Each short metric segment is clipped independently. This preserves the
    visible part of pitch markings that cross the camera near-plane (notably
    the touchline closest to a panned broadcast camera) and prevents projective
    jumps through Zc=0 from being drawn across the sky/stands.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return

    h, w = canvas.shape[:2]
    for i in range(len(pts) - 1):
        clipped3d = _clip_world_segment_to_near_plane(
            cam, pts[i], pts[i + 1], near_depth_m=near_depth_m
        )
        if clipped3d is None:
            continue

        a, b = clipped3d
        uv = cam.project_world(np.stack([a, b]))
        clipped2d = _clip_line_to_image(uv[0], uv[1], w, h)
        if clipped2d is None:
            continue

        q0, q1 = clipped2d
        p0 = tuple(np.round(q0).astype(np.int32))
        p1 = tuple(np.round(q1).astype(np.int32))
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)


def render_pitch_overlay(
    image_bgr: np.ndarray,
    cam: CameraState,
    thickness: int = 2,
    alpha: float = 0.85,
    line_color: Tuple[int, int, int] = (0, 255, 255),
    goal_color: Tuple[int, int, int] = (0, 128, 255),
) -> np.ndarray:
    """Project the complete canonical pitch model into raw broadcast pixels."""
    base = image_bgr.copy()
    overlay = image_bgr.copy()

    for pts, name in pitch_polylines(cam.pitch):
        color = goal_color if name in {"goal_post", "crossbar"} else line_color
        _draw_projected_polyline(overlay, cam, pts, color, thickness)

    out = cv2.addWeighted(overlay, float(alpha), base, float(1.0 - alpha), 0.0)
    label = f"Stage1 {cam.status.value}  frame={cam.frame_index}"
    cv2.putText(out, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(out, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_evidence_overlay(
    image_bgr: np.ndarray,
    keypoints: Optional[Mapping] = None,
    lines: Optional[Mapping] = None,
    *,
    draw_keypoint_hull: bool = True,
    coverage_metrics: Optional[Mapping] = None,
) -> np.ndarray:
    """Draw PnLCalib evidence and its spatial support.

    The convex hull is a QA aid only: it visualizes how widely keypoints support
    the camera estimate; it is not a pitch segmentation mask. Coordinates are in
    the original raw image pixel space.
    """
    out = image_bgr.copy()
    kp_pixels = []
    if keypoints:
        for key, value in keypoints.items():
            if not isinstance(value, Mapping) or "x" not in value or "y" not in value:
                continue
            x, y = float(value["x"]), float(value["y"])
            if np.isfinite(x) and np.isfinite(y):
                p = (int(round(x)), int(round(y)))
                kp_pixels.append([x, y])
                cv2.circle(out, p, 5, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(out, str(key), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

    if draw_keypoint_hull and len(kp_pixels) >= 3:
        hull = cv2.convexHull(np.asarray(kp_pixels, dtype=np.float32)).astype(np.int32)
        cv2.polylines(out, [hull], True, (255, 255, 0), 2, cv2.LINE_AA)

    line_count = 0
    if lines:
        for key, value in lines.items():
            if not isinstance(value, Mapping):
                continue
            req = ("x_1", "y_1", "x_2", "y_2")
            if not all(k in value for k in req):
                continue
            vals = [float(value[k]) for k in req]
            if np.all(np.isfinite(vals)):
                line_count += 1
                p1 = (int(round(vals[0])), int(round(vals[1])))
                p2 = (int(round(vals[2])), int(round(vals[3])))
                cv2.line(out, p1, p2, (255, 128, 0), 2, cv2.LINE_AA)
                cv2.putText(out, str(key), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 128, 0), 1, cv2.LINE_AA)

    metrics = coverage_metrics or {}
    hull_ratio = metrics.get("keypoint_image_convex_hull_area_ratio", metrics.get("image_convex_hull_area_ratio"))
    xspan = metrics.get("keypoint_image_x_span_ratio", metrics.get("image_x_span_ratio"))
    yspan = metrics.get("keypoint_image_y_span_ratio", metrics.get("image_y_span_ratio"))
    summary = f"PnL evidence: kp={len(kp_pixels)} lines={line_count}"
    if hull_ratio is not None:
        summary += f"  hull={float(hull_ratio):.3f}"
    if xspan is not None and yspan is not None:
        summary += f"  span=({float(xspan):.3f},{float(yspan):.3f})"
    cv2.putText(out, summary, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(out, summary, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_pitch_overlay(image_path: str, cam: CameraState, output_path: str) -> str:
    im = cv2.imread(str(image_path))
    if im is None:
        raise FileNotFoundError(image_path)
    out = render_pitch_overlay(im, cam)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), out):
        raise IOError(f"Could not write overlay to {output_path}")
    return str(output_path)


def _frustum_rays_world(cam: CameraState) -> Tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [
            [0.0, 0.0],
            [cam.image_width - 1.0, 0.0],
            [cam.image_width - 1.0, cam.image_height - 1.0],
            [0.0, cam.image_height - 1.0],
        ],
        dtype=np.float64,
    )
    origins, dirs = cam.world_ray(corners)
    return origins, dirs


def save_camera_3d_view(
    cam: CameraState,
    output_path: str,
    frustum_length_m: float = 25.0,
    optical_axis_length_m: float = 30.0,
) -> str:
    """3D QA: pitch, camera centre, optical axis and image-corner frustum."""
    p = cam.pitch
    x0, x1 = p.x_limits
    y0, y1 = p.y_limits

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for a, b, _ in pitch_segments(p):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])

    C = cam.camera_center_world_m
    ax.scatter([C[0]], [C[1]], [C[2]], s=60, label="camera")

    optical_world = cam.R_world_to_camera.T @ np.array([0.0, 0.0, 1.0])
    optical_world = optical_world / np.linalg.norm(optical_world)
    O_end = C + float(optical_axis_length_m) * optical_world
    ax.plot([C[0], O_end[0]], [C[1], O_end[1]], [C[2], O_end[2]], linewidth=2.5, label="optical axis")

    origins, dirs = _frustum_rays_world(cam)
    ends = origins + float(frustum_length_m) * dirs
    for end in ends:
        ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]], linewidth=1.2)
    for i in range(4):
        a = ends[i]
        b = ends[(i + 1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=1.0)

    # Also show where the centre pixel ray reaches the grass when it does.
    _, centre_dir = cam.world_ray(np.array([[cam.image_width / 2.0, cam.image_height / 2.0]]))
    dz = centre_dir[0, 2]
    if abs(dz) > 1e-9:
        lam = -C[2] / dz
        if lam > 0:
            q = C + lam * centre_dir[0]
            ax.scatter([q[0]], [q[1]], [q[2]], s=35, marker="x", label="centre-ray pitch hit")

    ax.set_xlim(min(x0 - 20.0, C[0] - 5.0), max(x1 + 20.0, C[0] + 5.0))
    ax.set_ylim(min(y0 - 20.0, C[1] - 5.0), max(y1 + 20.0, C[1] + 5.0))
    ax.set_zlim(0.0, max(25.0, float(C[2]) + 5.0))
    ax.set_xlabel("X goal-to-goal (m)")
    ax.set_ylabel("Y width (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_title(f"Camera world pose — frame {cam.frame_index} — {cam.status.value}")
    ax.legend(loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
