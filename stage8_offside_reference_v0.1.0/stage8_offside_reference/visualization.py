from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def render_longitudinal_qa(state: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render a simple 1D goalward-coordinate QA plot using matplotlib defaults."""
    import matplotlib.pyplot as plt

    ranking = state.get("opponent_ranking") or []
    ball = state.get("ball") or {}
    reference = state.get("reference") or {}

    fig, ax = plt.subplots(figsize=(10, 3.5))
    y = 1.0
    for row in ranking:
        q = row.get("goalward_q_m")
        if q is None:
            continue
        ax.scatter([q], [y])
        ax.text(q, y + 0.04, f"#{row.get('rank')} {row.get('track_id')}", rotation=35, ha="left", va="bottom")
        y += 0.12
    if ball.get("goalward_q_m") is not None:
        ax.scatter([ball["goalward_q_m"]], [0.6], marker="o")
        ax.text(ball["goalward_q_m"], 0.64, "ball", ha="center")
    if reference.get("goalward_q_m") is not None:
        ax.axvline(reference["goalward_q_m"], linestyle="--")
        ax.text(reference["goalward_q_m"], 0.45, f"reference: {reference.get('source')}", rotation=90, va="bottom")
    ax.set_xlabel("goalward coordinate q = sX (m)")
    ax.set_yticks([])
    ax.set_title(f"Stage 8 longitudinal QA | frame {state.get('frame_index')} | {state.get('status')}")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
