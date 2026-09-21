from __future__ import annotations

from pathlib import Path
import numpy as np

from ...pitch_draw import draw_metric_pitch

EDGES = (
    ("left_ear", "left_eye"), ("left_eye", "nose"), ("nose", "right_eye"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel"),
)


def _joint_dict(obs: dict) -> dict[str, np.ndarray]:
    out = {}
    for j in obs.get("joints", []):
        xyz = j.get("xyz_world_m")
        if xyz is not None:
            arr = np.asarray(xyz, dtype=float)
            if np.isfinite(arr).all():
                out[str(j.get("canonical_name"))] = arr
    return out


def save_topdown_world_pose(state: dict, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    out = Path(output_path).expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    selected = int(state.get("selected_frame", -1))
    fig, ax = plt.subplots(figsize=(12, 8)); draw_metric_pitch(ax)
    ax.set_title(f"Stage 4 v0.5.1 {state.get('backend')} - frame {selected}")
    ax.set_xlabel("Pitch X (m, goal-to-goal)"); ax.set_ylabel("Pitch Y (m)")
    for tr in state.get("tracks", []):
        obs = next((o for o in tr.get("observations", []) if int(o.get("frame_index", -1)) == selected), None)
        if obs is None or not obs.get("valid"):
            continue
        joints = _joint_dict(obs)
        if not joints:
            continue
        for a, b in EDGES:
            if a in joints and b in joints:
                ax.plot([joints[a][0], joints[b][0]], [joints[a][1], joints[b][1]], linewidth=1.1)
        arr = np.stack(list(joints.values()))
        ax.scatter(arr[:, 0], arr[:, 1], s=14)
        c = np.nanmean(arr[:, :2], axis=0)
        ax.text(c[0] + .25, c[1] + .25, str(tr.get("track_id")), fontsize=8)
        translation = obs.get("translation", {})
        for key, marker, color, label in (
            ("ground_hit_world_m", "x", "black", "RTMW foot / pitch hit"),
            ("sam_root_world_m", "o", "red", "SAM translation"),
            ("refined_root_world_m", "^", "blue", "Output translation"),
        ):
            point = translation.get(key)
            if point is not None:
                ax.scatter(point[0], point[1], marker=marker, color=color, s=50, label=label)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique: ax.legend(unique.values(), unique.keys(), loc="upper left")
    ax.set_aspect("equal", adjustable="box"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def save_selected_frame_overlay(state: dict, image_bgr: np.ndarray, output_path: str | Path) -> Path:
    import cv2
    out = Path(output_path).expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(image_bgr).copy(); selected = int(state.get("selected_frame", -1))
    for tr in state.get("tracks", []):
        obs = next((o for o in tr.get("observations", []) if int(o.get("frame_index", -1)) == selected), None)
        if obs is None or not obs.get("valid"):
            continue
        pts = {}
        for j in obs.get("joints", []):
            uv = j.get("reprojected_uv_px")
            if uv is None: continue
            uv = np.asarray(uv, dtype=float)
            if np.isfinite(uv).all(): pts[str(j.get("canonical_name"))] = uv
        for a,b in EDGES:
            if a in pts and b in pts:
                cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), (255,255,255), 1, cv2.LINE_AA)
        for uv in pts.values():
            cv2.circle(img, tuple(np.round(uv).astype(int)), 2, (255,255,255), -1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)
    return out
