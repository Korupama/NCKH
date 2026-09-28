from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np

from .adapters import (
    bbox_from_stage3_observation,
    keypoints_from_stage3_observation,
    selected_stage3_observation,
    stage3_track_map,
)
from .geometry import stage4_world_points
from .projection import project_world_points


PALETTE = {
    "OFFSIDE_POSITION": (40, 40, 230),    # BGR red-ish
    "ONSIDE": (70, 190, 70),             # green
    "TOUCHER_EXCLUDED": (225, 150, 40),   # blue-ish
    "UNAVAILABLE": (150, 150, 150),
    "DEFENDER": (20, 190, 230),
    "REFERENCE": (230, 80, 220),
    "BALL": (240, 240, 240),
    "REFEREE": (190, 110, 220),
    "UNKNOWN_TEAM": (135, 135, 135),
    "INACTIVE": (95, 95, 95),
}


def _ensure_cv2():
    import cv2
    return cv2


def load_frame_with_source(*, image_path: Optional[str] = None, video_path: Optional[str] = None, frame_index: int = 0, fallback_size=(1280, 720)):
    """Load an image-space background and return explicit provenance.

    Stage 9 never attempts to clean or reinterpret pixels. If the supplied source
    already contains skeletons/boxes, they will remain baked into the background.
    """
    cv2 = _ensure_cv2()
    if image_path:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img, {
                "kind": "IMAGE",
                "path": str(Path(image_path)),
                "frame_index": int(frame_index),
                "warning": None,
            }
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, img = cap.read()
        cap.release()
        if ok and img is not None:
            return img, {
                "kind": "VIDEO",
                "path": str(Path(video_path)),
                "frame_index": int(frame_index),
                "warning": None,
            }
    w, h = fallback_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (28, 38, 30)
    cv2.putText(canvas, "Stage 9 demo: no source frame supplied", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
    return canvas, {
        "kind": "FALLBACK_CANVAS",
        "path": None,
        "frame_index": int(frame_index),
        "warning": "NO_SOURCE_FRAME_AVAILABLE",
    }


def load_frame(*, image_path: Optional[str] = None, video_path: Optional[str] = None, frame_index: int = 0, fallback_size=(1280, 720)):
    """Backward-compatible image-only wrapper."""
    return load_frame_with_source(
        image_path=image_path, video_path=video_path, frame_index=frame_index, fallback_size=fallback_size
    )[0]


def _clip_point(pt, width, height):
    return int(max(0, min(width - 1, round(float(pt[0]))))), int(max(0, min(height - 1, round(float(pt[1])))))


def _bbox_from_projected(points: np.ndarray, width: int, height: int) -> Optional[List[int]]:
    if points.size == 0:
        return None
    good = np.isfinite(points).all(axis=1)
    pts = points[good]
    if len(pts) == 0:
        return None
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    pad_x = max(8.0, (x2 - x1) * 0.15)
    pad_y = max(8.0, (y2 - y1) * 0.08)
    return [
        int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)),
        int(min(width - 1, x2 + pad_x)), int(min(height - 1, y2 + pad_y)),
    ]


def _sanitize_bbox(bbox: Optional[List[float]], width: int, height: int) -> Optional[List[int]]:
    if bbox is None or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    x1=max(0.0,min(width-1.0,x1)); x2=max(0.0,min(width-1.0,x2))
    y1=max(0.0,min(height-1.0,y1)); y2=max(0.0,min(height-1.0,y2))
    if x2-x1 < 2.0 or y2-y1 < 2.0:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def resolve_track_visual_geometry(
    *, track_id: str, frame_index: int, width: int, height: int,
    stage3_tracks: Mapping[str, Mapping[str, Any]], stage4_tracks: Mapping[str, Mapping[str, Any]],
    stage1: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve only grounded screen geometry; never fabricate coordinates."""
    bbox = None
    kp2d: List[Dict[str, Any]] = []
    source = None
    s3t = stage3_tracks.get(track_id)
    if s3t is not None:
        obs3 = selected_stage3_observation(s3t, frame_index)
        bbox = _sanitize_bbox(bbox_from_stage3_observation(obs3), width, height)
        kp2d = keypoints_from_stage3_observation(obs3)
        if bbox is not None:
            source = "STAGE3_BBOX"

    projected = None
    s4t = stage4_tracks.get(track_id)
    if s4t is not None and stage1 is not None:
        world_pts = stage4_world_points(s4t, frame_index)
        if world_pts:
            xyz = [p["xyz_world_m"] for p in world_pts]
            projected = project_world_points(stage1, xyz, distort=True)
            if bbox is None:
                bbox = _sanitize_bbox(_bbox_from_projected(projected, width, height), width, height)
                if bbox is not None:
                    source = "STAGE4_PROJECTED_BBOX"

    return {"bbox": bbox, "keypoints_2d": kp2d, "projected": projected, "source": source}


def _defender_visible(row: Mapping[str, Any], show_all_defenders: bool) -> bool:
    if show_all_defenders:
        return True
    try:
        rank = int(row.get("rank"))
    except (TypeError, ValueError):
        return False
    return rank in (1, 2)


def _overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _draw_label(img, bbox, text, color, occupied):
    cv2 = _ensure_cv2()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    scale = 0.52
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    lw, lh = tw + 12, th + baseline + 9
    h, w = img.shape[:2]
    candidates = [
        (x1, y1 - lh),
        (x2 + 6, y1),
        (x1, y2 + 5),
        (x1 - lw - 6, y1),
    ]
    chosen = None
    for lx, ly in candidates:
        lx = max(0, min(w - lw - 1, lx))
        ly = max(0, min(h - lh - 1, ly))
        rect = (lx, ly, lx + lw, ly + lh)
        if not any(_overlap(rect, old) for old in occupied):
            chosen = rect
            break
    if chosen is None:
        lx = max(0, min(w - lw - 1, x1))
        ly = max(0, min(h - lh - 1, y1 - lh - 8 * len(occupied)))
        chosen = (lx, ly, lx + lw, ly + lh)
    occupied.append(chosen)
    lx, ly, rx, by = chosen
    cv2.rectangle(img, (lx, ly), (rx, by), color, -1)
    cv2.putText(img, text, (lx + 6, ly + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), thickness, cv2.LINE_AA)
    anchor = ((x1 + x2)//2, y1)
    target = (min(max(anchor[0], lx), rx), min(max(anchor[1], ly), by))
    cv2.line(img, anchor, target, color, 1, cv2.LINE_AA)


def _project_reference_line(stage1, x_ref, width, height):
    if stage1 is None or x_ref is None:
        return None
    ys = np.linspace(-36.0, 36.0, 80)
    pts = np.column_stack([np.full_like(ys, float(x_ref)), ys, np.zeros_like(ys)])
    uv = project_world_points(stage1, pts, distort=True)
    good = np.isfinite(uv).all(axis=1)
    uv = uv[good]
    if len(uv) < 2:
        return None
    return [(_clip_point(p, width, height)) for p in uv]


def render_overlay(
    frame,
    stage9: Mapping[str, Any],
    *,
    stage4: Optional[Mapping[str, Any]] = None,
    stage3: Optional[Mapping[str, Any]] = None,
    stage1: Optional[Mapping[str, Any]] = None,
    show_reference: bool = True,
    show_defenders: bool = True,
    show_skeleton: bool = False,
    show_labels: bool = True,
    show_all_defenders: bool = False,
):
    cv2 = _ensure_cv2()
    img = frame.copy()
    h, w = img.shape[:2]
    frame_index = int(stage9.get("frame_index") or 0)
    stage4_tracks = {str(t.get("track_id")): t for t in ((stage4 or {}).get("tracks") or []) if isinstance(t, dict) and t.get("track_id") is not None}
    stage3_tracks = stage3_track_map(stage3 or {})

    # Reference ground trace.
    ref = stage9.get("reference") or {}
    if show_reference and ref.get("X_world_m") is not None and stage1:
        poly = _project_reference_line(stage1, ref.get("X_world_m"), w, h)
        if poly and len(poly) >= 2:
            cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], False, PALETTE["REFERENCE"], 4, cv2.LINE_AA)
            px, py = poly[len(poly)//2]
            cv2.putText(img, f"REF X={float(ref['X_world_m']):.2f}m", (max(8, px+8), max(25, py-8)), cv2.FONT_HERSHEY_SIMPLEX, .65, PALETTE["REFERENCE"], 2, cv2.LINE_AA)

    occupied_labels = []
    draw_rows = []
    for row in stage9.get("attackers") or []:
        draw_rows.append((row, row.get("label") or "UNAVAILABLE"))
    if show_defenders:
        for row in stage9.get("opponents") or []:
            if _defender_visible(row, show_all_defenders):
                draw_rows.append((row, "DEFENDER"))
        # Contextual rows stay out of the image by default. Their semantic state
        # remains available in the side panel/API, but Stage 9 must not invent
        # image-space positions for them.

    for row, label in draw_rows:
        tid = str(row.get("track_id"))
        color = PALETTE.get(label, PALETTE["UNAVAILABLE"])
        visual = resolve_track_visual_geometry(
            track_id=tid, frame_index=frame_index, width=w, height=h,
            stage3_tracks=stage3_tracks, stage4_tracks=stage4_tracks, stage1=stage1,
        )
        bbox = visual["bbox"]
        kp2d = visual["keypoints_2d"]
        projected = visual["projected"]

        # Critical policy: no hash/random/fabricated screen coordinates. If
        # image geometry cannot be grounded in Stage 3 or Stage 4+Stage 1, the
        # semantic ON/OFF result remains in the web side panel/API only.
        if bbox is None:
            continue

        if show_skeleton:
            if projected is not None:
                for p in projected:
                    if np.isfinite(p).all():
                        cv2.circle(img, _clip_point(p, w, h), 3, color, -1, cv2.LINE_AA)
            elif kp2d:
                for p in kp2d:
                    cv2.circle(img, _clip_point((p["x"], p["y"]), w, h), 2, color, -1, cv2.LINE_AA)

        if show_labels:
            if label == "OFFSIDE_POSITION":
                flag = "OFFSIDE"
            elif label == "ONSIDE":
                flag = "ONSIDE"
            elif label == "TOUCHER_EXCLUDED":
                flag = "PASSER"
            elif label == "DEFENDER":
                rank = row.get("rank")
                flag = "LAST DEF" if rank == 1 else "2ND LAST" if rank == 2 else f"DEF #{rank}"
            else:
                flag = "ON?"
            delta = row.get("delta_q_m")
            suffix = f" {float(delta):+.2f}m" if delta is not None else ""
            _draw_label(img, bbox, f"{tid} | {flag}{suffix}", color, occupied_labels)

        # Highlight chosen goalward anchor.
        anchor = ((row.get("geometry") or {}).get("anchor") if label != "DEFENDER" else row.get("anchor")) or {}
        xyz = anchor.get("xyz_world_m") if isinstance(anchor, dict) else None
        if xyz is not None and stage1 is not None:
            uv = project_world_points(stage1, [xyz], distort=True)[0]
            if np.isfinite(uv).all():
                cv2.circle(img, _clip_point(uv, w, h), 8, color, 2, cv2.LINE_AA)

    # Ball marker when Stage-8/6 metric center is available.
    ball = stage9.get("ball") or {}
    ball_xyz = ball.get("center_xyz_world_m") if isinstance(ball, dict) else None
    if ball_xyz is not None and stage1 is not None:
        try:
            uv = project_world_points(stage1, [ball_xyz], distort=True)[0]
            if np.isfinite(uv).all():
                bp = _clip_point(uv, w, h)
                cv2.circle(img, bp, 8, (20,20,20), 2, cv2.LINE_AA)
                cv2.circle(img, bp, 5, PALETTE["BALL"], -1, cv2.LINE_AA)
                cv2.putText(img, "BALL", (bp[0]+10, max(18,bp[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, .48, PALETTE["BALL"], 2, cv2.LINE_AA)
        except Exception:
            pass

    # Header banner.
    mode = stage9.get("mode", "BEST_EFFORT_DEMO")
    off_n = sum(1 for r in stage9.get("attackers") or [] if r.get("label") == "OFFSIDE_POSITION")
    on_n = sum(1 for r in stage9.get("attackers") or [] if r.get("label") == "ONSIDE")
    cv2.rectangle(img, (0, 0), (w, 58), (18, 18, 18), -1)
    cv2.putText(img, f"Stage 9 | {mode} | ON {on_n} | OFF {off_n}", (18, 37), cv2.FONT_HERSHEY_SIMPLEX, .82, (245,245,245), 2, cv2.LINE_AA)
    return img


def encode_jpeg(image, quality: int = 90) -> bytes:
    cv2 = _ensure_cv2()
    ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Could not encode JPEG")
    return bytes(buf)


def save_overlay(path: str | Path, image) -> Path:
    cv2 = _ensure_cv2()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), image):
        raise RuntimeError(f"Could not write {p}")
    return p
