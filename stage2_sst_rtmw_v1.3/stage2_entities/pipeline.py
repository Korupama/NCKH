from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import json

from .contracts import EntityTrackState, ReplayContext
from .tracking import build_entity_track_state
from .visualization import render_selected_frame, render_tracking_video


def finalize_stage2(
    *,
    context: ReplayContext,
    perception_manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    render_video: bool = True,
    show_excluded_debug: bool = False,
    max_assignment_cost: float = 0.92,
    max_gap: int = 6,
    use_temporal_rescue: bool = True,
    rescue_max_assignment_cost: float = 0.78,
) -> EntityTrackState:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = build_entity_track_state(
        perception_manifest, context, out,
        max_assignment_cost=max_assignment_cost,
        max_gap=max_gap,
        use_temporal_rescue=use_temporal_rescue,
        rescue_max_assignment_cost=rescue_max_assignment_cost,
    )

    entity_path = out / "stage2_entity_tracks.json"
    state.artifacts["entity_track_state"] = str(entity_path.resolve())

    handoff_path = out / "stage3_handoff.json"
    handoff_path.write_text(json.dumps(state.stage3_handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    state.artifacts["stage3_handoff"] = str(handoff_path.resolve())

    if Path(context.video_path).is_file():
        selected = render_selected_frame(context, state, out / "stage2_selected_frame.png")
        debug = render_selected_frame(
            context, state, out / "stage2_selected_frame_debug_all_humans.png", show_excluded=True
        )
        state.artifacts["selected_frame_visual"] = str(selected.resolve())
        state.artifacts["selected_frame_debug_visual"] = str(debug.resolve())
        if render_video:
            video = render_tracking_video(context, state, out / "stage2_tracking_window.mp4")
            state.artifacts["tracking_video"] = str(video.resolve())
    state.save_json(entity_path)
    return state
