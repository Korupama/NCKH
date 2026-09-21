from __future__ import annotations

from pathlib import Path
import numpy as np

from ...pitch_draw import draw_metric_pitch


def save_topdown_world_pose(state: dict, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    selected = int(state.get("selected_frame", -1))
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_metric_pitch(ax)
    ax.set_title(f"Stage 4 v0.4 world-grounded pose - frame {selected}")
    ax.set_xlabel("Pitch X (m, goal-to-goal)")
    ax.set_ylabel("Pitch Y (m)")
    for tr in state.get("tracks", []):
        obs = next((o for o in tr.get("observations", []) if int(o.get("frame_index", -1)) == selected), None)
        if obs is None or not obs.get("valid"):
            continue
        pts = [j.get("xyz_world_m") for j in obs.get("joints", []) if j.get("xyz_world_m") is not None]
        if not pts:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        ax.scatter(arr[:, 0], arr[:, 1], s=14, label=str(tr.get("track_id")))
        center = np.nanmean(arr[:, :2], axis=0)
        ax.text(center[0] + 0.3, center[1] + 0.3, str(tr.get("track_id")), fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.2})
    if any(tr.get("selected_frame_status") == "VALID" for tr in state.get("tracks", [])):
        ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
