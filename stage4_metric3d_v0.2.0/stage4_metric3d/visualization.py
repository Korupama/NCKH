from __future__ import annotations

from pathlib import Path
from typing import Mapping
import numpy as np

from .pitch_draw import draw_metric_pitch


def save_topdown_selected_frame(state: Mapping[str, object], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    # Draw pitch context before tracks, so every Stage 4 QA image is legible
    # without separately opening camera-calibration or pitch metadata.
    draw_metric_pitch(ax)
    ax.set_xlabel("Pitch X (m)")
    ax.set_ylabel("Pitch Y (m)")
    ax.set_title(f"Stage 4 metric 3D selected frame {state.get('replay_context', {}).get('selected_frame')}")
    for item in state.get("selected_frame_poses", []):
        obs = item.get("observation", {})
        joints = obs.get("metric_pose23", [])
        pts = []
        for j in joints:
            xyz = j.get("xyz_world_m")
            if xyz is not None:
                pts.append(xyz)
        if pts:
            arr = np.asarray(pts, dtype=float)
            track_id = str(item.get("track_id"))
            ax.scatter(arr[:, 0], arr[:, 1], s=12, label=track_id, zorder=10)
            label_xy = np.nanmean(arr[:, :2], axis=0)
            ax.text(
                label_xy[0] + 0.4,
                label_xy[1] + 0.4,
                track_id,
                fontsize=8,
                color="#222222",
                zorder=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )
    if state.get("selected_frame_poses"):
        ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
