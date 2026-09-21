from __future__ import annotations

import argparse
import json
from pathlib import Path

from ball_localization.geometry import HybridGeometryConfig, TemporalRefinementConfig
from ball_localization.pipeline import refine_existing_state


def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply Stage 6 v0.4 temporal diameter/range refinement to an existing ball_trajectory_state.json without rerunning YOLO"
    )
    p.add_argument("--state-json", required=True)
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--diameter-second-diff-weight", type=float, default=8.0)
    p.add_argument("--diameter-first-diff-weight", type=float, default=0.10)
    p.add_argument("--max-gap-frames", type=int, default=3)
    p.add_argument("--range-sigma-fraction", type=float, default=0.08)
    p.add_argument("--position-second-diff-sigma-m", type=float, default=0.75)
    p.add_argument("--position-third-diff-sigma-m", type=float, default=0.50)
    p.add_argument("--pitch-margin-m", type=float, default=6.0)
    p.add_argument("--max-height-m", type=float, default=30.0)
    p.add_argument("--hybrid-config-json", help="Frozen v0.4.4 hybrid config; uses cached candidates and never reruns YOLO")
    a = p.parse_args()
    cfg = TemporalRefinementConfig(
        diameter_second_diff_weight=a.diameter_second_diff_weight,
        diameter_first_diff_weight=a.diameter_first_diff_weight,
        max_gap_frames=a.max_gap_frames,
        range_sigma_fraction=a.range_sigma_fraction,
        position_second_diff_sigma_m=a.position_second_diff_sigma_m,
        position_third_diff_sigma_m=a.position_third_diff_sigma_m,
        pitch_margin_m=a.pitch_margin_m,
        max_height_m=a.max_height_m,
    )
    state = refine_existing_state(
        state_json=a.state_json,
        stage1_root=a.stage1_root,
        output_dir=a.output_dir,
        temporal_config=cfg,
        hybrid_config=HybridGeometryConfig.from_mapping((json.loads(Path(a.hybrid_config_json).read_text(encoding="utf-8")).get("best") or {}).get("config")) if a.hybrid_config_json else None,
    )
    print(json.dumps({
        "status": state.status,
        "stage6_version": state.stage6_version,
        "selected_frame_ball": state.selected_frame_ball,
        "temporal_refinement": state.diagnostics.get("temporal_refinement"),
        "artifacts": state.artifacts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
