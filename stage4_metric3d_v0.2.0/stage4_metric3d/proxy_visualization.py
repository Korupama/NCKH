from __future__ import annotations

from pathlib import Path
from typing import Mapping
import numpy as np

from .pitch_draw import draw_metric_pitch


GROUP_COLORS = {
    "head": "#7b1fa2",
    "shoulder": "#d32f2f",
    "arm": "#f57c00",
    "hip": "#1976d2",
    "knee": "#0097a7",
    "ankle": "#388e3c",
    "foot": "#689f38",
}


def save_proxy_topdown_selected_frame(state: Mapping[str, object], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pitch = ((state.get("coordinate_frame") or {}).get("pitch") or {})
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_metric_pitch(
        ax,
        pitch_length=float(pitch.get("length_m", 105.0)),
        pitch_width=float(pitch.get("width_m", 68.0)),
    )
    frame = (state.get("replay_context") or {}).get("selected_frame")
    ax.set_title(f"Stage 4 v0.3.1 raw height-plane proxies (audit only) — frame {frame}")
    ax.set_xlabel("Pitch X (m, goal-to-goal)")
    ax.set_ylabel("Pitch Y (m)")
    used_groups = set()
    ground_label_used = False
    for track in state.get("selected_frame_proxies", []):
        observation = track.get("observation") or {}
        points = []
        for point in observation.get("raw_height_plane_proxies_23", observation.get("body_proxies_23", [])):
            xyz = point.get("xyz_proxy_world_m")
            if xyz is None:
                continue
            group = str(point.get("anatomical_group"))
            color = GROUP_COLORS.get(group, "#444444")
            label = group if group not in used_groups else None
            used_groups.add(group)
            ax.scatter([xyz[0]], [xyz[1]], s=14, color=color, label=label, zorder=10)
            points.append(xyz)
        if points:
            ground = observation.get("ground_anchor") or {}
            ground_xyz = ground.get("xyz_ground_m")
            if ground_xyz is not None and ground.get("status") in {"VALID", "DEGRADED"}:
                ax.scatter(
                    [ground_xyz[0]],
                    [ground_xyz[1]],
                    marker="D",
                    s=34,
                    color="#111111",
                    label=None if ground_label_used else "safe ground anchor",
                    zorder=12,
                )
                ground_label_used = True
                center = np.asarray(ground_xyz[:2], dtype=float)
            else:
                center = np.nanmean(np.asarray(points, dtype=float)[:, :2], axis=0)
            ax.text(
                center[0] + 0.35,
                center[1] + 0.35,
                str(track.get("track_id")),
                fontsize=8,
                zorder=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
            )
    if used_groups or ground_label_used:
        ax.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
